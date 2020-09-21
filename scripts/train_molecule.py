import sys
from os import path as osp
import time
from math import sqrt
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import transformers

import forge
from forge import flags
import forge.experiment_tools as fet
from copy import deepcopy
from collections import defaultdict
import deepdish as dd
from tqdm import tqdm

from eqv_transformer.train_tools import (
    log_tensorboard,
    parse_reports,
    print_reports,
    log_reports,
    load_checkpoint,
    save_checkpoint,
    ExponentialMovingAverage,
    get_component,
    nested_to,
    param_count,
    get_component,
    get_average_norm,
    param_count,
    parameter_analysis,
)

from eqv_transformer.molecule_predictor import MoleculePredictor
from eqv_transformer.multihead_neural import MultiheadLinear
from lie_conv.utils import Pass, Expression
from lie_conv.masked_batchnorm import MaskBatchNormNd


if torch.cuda.is_available():
    device = "cuda"
    # device = "cpu"
else:
    device = "cpu"

#####################################################################################################################
# Command line flags
#####################################################################################################################
# Directories
flags.DEFINE_string("data_dir", "data/", "Path to data directory")
flags.DEFINE_string(
    "results_dir", "checkpoints/", "Top directory for all experimental results."
)

# Configuration files to load
flags.DEFINE_string(
    "data_config", "configs/molecule/qm9_data.py", "Path to a data config file."
)
flags.DEFINE_string(
    "model_config",
    "configs/molecule/set_transformer.py",
    "Path to a model config file.",
)
# Job management
flags.DEFINE_string("run_name", "test", "Name of this job and name of results folder.")
flags.DEFINE_boolean("resume", False, "Tries to resume a job if True.")

# Logging
flags.DEFINE_integer(
    "report_loss_every", 500, "Number of iterations between reporting minibatch loss."
)
flags.DEFINE_integer(
    "evaluate_every", 10000, "Number of iterations between reporting validation loss."
)
flags.DEFINE_integer(
    "save_check_points",
    10,
    "frequency with which to save checkpoints, in number of epochs.",
)
flags.DEFINE_boolean("log_train_values", True, "Logs train values if True.")
flags.DEFINE_float(
    "ema_alpha", 0.99, "Alpha coefficient for exponential moving average of train logs."
)

# Optimization
flags.DEFINE_integer("train_epochs", 500, "Maximum number of training epochs.")
flags.DEFINE_integer("batch_size", 90, "Mini-batch size.")
flags.DEFINE_float("learning_rate", 1e-5, "SGD learning rate.")
flags.DEFINE_float("beta1", 0.5, "Adam Beta 1 parameter")
flags.DEFINE_float("beta2", 0.9, "Adam Beta 2 parameter")
flags.DEFINE_string(
    "lr_schedule", "none", "What learning rate schedule to use. Options: cosine, none",
)
flags.DEFINE_boolean(
    "parameter_count", False, "If True, print model parameter count and exit"
)
flags.DEFINE_boolean("debug", False, "Enable additional telemetry for debugging")
flags.DEFINE_boolean(
    "init_activations",
    False,
    "produce initialisation activation histograms the activations of specified modules through training",
)
flags.DEFINE_boolean("profile_model", False, "Run profiling code on model and exit")

#####################################################################################################################


# @forge.debug_on(Exception)
def main():
    # Parse flags
    config = forge.config()

    # Load data
    dataloaders, num_species, charge_scale, ds_stats, data_name = fet.load(
        config.data_config, config=config
    )

    config.num_species = num_species
    config.charge_scale = charge_scale
    config.ds_stats = ds_stats

    # Load model
    model, model_name = fet.load(config.model_config, config)
    model.to(device)

    num_params = param_count(model)
    if config.parameter_count:
        print(model)
        print("============================================================")
        print(f"{model_name} parameters: {num_params:.5e}")
        print("============================================================")
        from torchsummary import summary

        data = next(iter(dataloaders["train"]))
        data = {k: v.to(device) for k, v in data.items()}
        print(summary(model.predictor, data, batch_size=config.batch_size,))
        # fet.print_flags()
        model(data).loss.backward()
        sys.exit(0)
    else:
        print(f"{model_name} parameters: {num_params:.5e}")

    config.charge_scale = float(config.charge_scale.numpy())
    config.ds_stats = [float(stat.numpy()) for stat in config.ds_stats]

    # Prepare environment
    run_name = (
        config.run_name
        + "_bs"
        + str(config.batch_size)
        + "_lr"
        + str(config.learning_rate)
    )

    if config.batch_fit != 0:
        run_name += "_bf" + str(config.batch_fit)

    if config.lr_schedule != "none":
        run_name += "_" + config.lr_schedule

    # if config.block_norm != "none":
    #     run_name += "_" + config.block_norm

    # if config.kernel_norm != "none":
    #     run_name += "_" + config.kernel_norm

    results_folder_name = osp.join(data_name, model_name, run_name,)

    logdir = osp.join(config.results_dir, results_folder_name.replace(".", "_"))
    logdir, resume_checkpoint = fet.init_checkpoint(
        logdir, config.data_config, config.model_config, config.resume
    )

    checkpoint_name = osp.join(logdir, "model.ckpt")

    # Print flags
    fet.print_flags()

    # Setup optimizer
    model_params = model.predictor.parameters()

    opt_learning_rate = config.learning_rate
    model_opt = torch.optim.Adam(
        model_params, lr=opt_learning_rate, betas=(config.beta1, config.beta2)
    )
    # model_opt = torch.optim.SGD(model_params, lr=opt_learning_rate)

    # Cosine annealing learning rate
    if config.lr_schedule == "cosine_warmup":
        num_warmup_epochs = int(0.05 * config.train_epochs)
        num_warmup_steps = num_warmup_epochs
        num_training_steps = config.train_epochs
        lr_schedule = transformers.get_cosine_schedule_with_warmup(
            model_opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        # b = []
        # for i in range(num_training_steps * len(dataloaders["train"])):
        #     b.append(model_opt.param_groups[0]["lr"])
        #     lr_schedule.step(i / len(dataloaders["train"]))
        # for i in b:
        #     print(i)
        # print(b)
        # print(b[8000])
        # print(num_training_steps)
        # print(max(zip(b, range(len(b)))))
        # sys.exit(0)

    elif config.lr_schedule == "quadratic_warmup":
        lr_sched = lambda e: min(e / (0.01 * config.train_epochs), 1) * (
            1.0
            / sqrt(
                1.0 + 10000.0 * (e / config.train_epochs)
            )  # finish at 1/100 of initial lr
        )
        lr_schedule = optim.lr_scheduler.LambdaLR(model_opt, lr_sched)
    elif config.lr_schedule == "none":
        lr_sched = lambda e: 1.0
        lr_schedule = optim.lr_scheduler.LambdaLR(model_opt, lr_sched)
    else:
        raise ValueError(
            f"{config.lr_schedule} is not a recognised learning rate schedule"
        )

    # Try to restore model and optimizer from checkpoint
    if resume_checkpoint is not None:
        start_epoch = load_checkpoint(resume_checkpoint, model, model_opt, lr_schedule)
    else:
        start_epoch = 1

    train_iter = (start_epoch - 1) * (
        len(dataloaders["train"].dataset) // config.batch_size
    ) + 1

    print("Starting training at epoch = {}, iter = {}".format(start_epoch, train_iter))

    # Do model profiling if desired
    if config.profile_model:
        # import torch.autograd.profiler as profiler

        # model.to("cpu")

        # data = next(iter(dataloaders["train"]))

        # data = {k: v.to(device) for k, v in data.items()}

        # # with torch.cuda.profiler.profile():
        # #     outputs = model(data, compute_loss=True)
        # #     outputs.loss.backward()
        # #     with torch.autograd.profiler.emit_nvtx() as prof:
        # #         outputs = model(data, compute_loss=True)
        # #         outputs.loss.backward()

        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        #     with profiler.record_function("model_inference"):
        #         outputs = model(data, compute_loss=True)
        #         outputs.loss.backward()

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        parameter_analysis(model)

        sys.exit(0)

    # Setup tensorboard writing
    summary_writer = SummaryWriter(logdir)

    report_all = defaultdict(list)
    # Saving model at epoch 0 before training
    print("saving model at epoch 0 before training ... ")
    save_checkpoint(checkpoint_name, 0, model, model_opt, lr_schedule, 0.0)
    print("finished saving model at epoch 0 before training")

    if (
        config.debug
        and config.model_config == "configs/dynamics/eqv_transformer_model.py"
    ):
        model_components = (
            [(0, [], "embedding_layer")]
            + list(
                chain.from_iterable(
                    (
                        (k, [], f"ema_{k}"),
                        (
                            k,
                            ["ema", "kernel", "location_kernel"],
                            f"ema_{k}_location_kernel",
                        ),
                        (
                            k,
                            ["ema", "kernel", "feature_kernel"],
                            f"ema_{k}_feature_kernel",
                        ),
                    )
                    for k in range(1, config.num_layers + 1)
                )
            )
            + [(config.num_layers + 2, [], "output_mlp")]
        )  # components to track for debugging
        grad_flows = []

    if config.init_activations:
        activation_tracked = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, Expression)
            | isinstance(module, nn.Linear)
            | isinstance(module, MultiheadLinear)
            | isinstance(module, MaskBatchNormNd)
        ]
        activations = {}

        def save_activation(name, mod, inpt, otpt):
            # print(name)
            # print(len(inpt))
            # print(len(otpt))

            if isinstance(inpt, tuple):
                if isinstance(inpt[0], list):
                    activations[name + "_inpt"] = inpt[0][1].detach().cpu()
                else:
                    if len(inpt) == 1:
                        activations[name + "_inpt"] = inpt[0].detach().cpu()
                    else:
                        activations[name + "_inpt"] = inpt[1].detach().cpu()
            else:
                activations[name + "_inpt"] = inpt.detach().cpu()

            if isinstance(otpt, tuple):
                if isinstance(otpt[0], list):
                    activations[name + "_otpt"] = otpt[0][1].detach().cpu()
                else:
                    if len(otpt) == 1:
                        activations[name + "_otpt"] = otpt[0].detach().cpu()
                    else:
                        activations[name + "_otpt"] = otpt[1].detach().cpu()
            else:
                activations[name + "_otpt"] = otpt.detach().cpu()

        for name, tracked_module in activation_tracked:
            tracked_module.register_forward_hook(partial(save_activation, name))

    # Training
    start_t = time.perf_counter()
    if config.log_train_values:
        train_emas = ExponentialMovingAverage(alpha=config.ema_alpha, debias=True)

    iters_per_epoch = len(dataloaders["train"])
    for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):
        model.train()

        for batch_idx, data in enumerate(dataloaders["train"]):
            data = {k: v.to(device) for k, v in data.items()}

            model_opt.zero_grad()
            outputs = model(data, compute_loss=True)
            outputs.loss.backward()
            model_opt.step()

            if config.init_activations:
                for name, activation in activations.items():
                    print(name)
                    summary_writer.add_histogram(
                        f"activations/{name}", activation.numpy(), 0
                    )

                sys.exit(0)

            if config.log_train_values:
                reports = parse_reports(outputs.reports)
                if batch_idx % config.report_loss_every == 0:
                    log_tensorboard(summary_writer, train_iter, reports, "train/")
                    report_all = log_reports(report_all, train_iter, reports, "train")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(dataloaders["train"].dataset) // config.batch_size,
                        prefix="train",
                    )

            # Logging
            if batch_idx % config.evaluate_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_mae = 0.0
                    for data in dataloaders["valid"]:
                        data = {k: v.to(device) for k, v in data.items()}
                        outputs = model(data, compute_loss=True)
                        valid_mae = valid_mae + outputs.mae

                outputs["reports"].valid_mae = valid_mae / len(dataloaders["valid"])

                reports = parse_reports(outputs.reports)

                log_tensorboard(summary_writer, train_iter, reports, "valid")
                report_all = log_reports(report_all, train_iter, reports, "valid")
                print_reports(
                    reports,
                    start_t,
                    epoch,
                    batch_idx,
                    len(dataloaders["train"].dataset) // config.batch_size,
                    prefix="valid",
                )

            train_iter += 1

            # Step the LR schedule
            lr_schedule.step(train_iter / iters_per_epoch)

            # Track stuff for debugging
            if config.debug:
                model.eval()
                with torch.no_grad():
                    curr_params = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }

                    # updates norm
                    if train_iter == 1:
                        update_norm = 0
                    else:
                        update_norms = []
                        for (k_prev, v_prev), (k_curr, v_curr) in zip(
                            prev_params.items(), curr_params.items()
                        ):
                            assert k_prev == k_curr
                            if (
                                "tracked" not in k_prev
                            ):  # ignore batch norm tracking. TODO: should this be ignored? if not, fix!
                                update_norms.append((v_curr - v_prev).norm(1).item())
                        update_norm = sum(update_norms)

                    # gradient norm
                    grad_norm = 0
                    for p in model.parameters():
                        try:
                            grad_norm += p.grad.norm(1)
                        except AttributeError:
                            pass

                    # weights norm
                    if (
                        config.model_config
                        == "configs/dynamics/eqv_transformer_model.py"
                    ):
                        model_norms = {}
                        for comp_name in model_components:
                            comp = get_component(model.predictor.net, comp_name)
                            norm = get_average_norm(comp)
                            model_norms[comp_name[2]] = norm

                        log_tensorboard(
                            summary_writer,
                            train_iter,
                            model_norms,
                            "debug/avg_model_norms/",
                        )

                    log_tensorboard(
                        summary_writer,
                        train_iter,
                        {
                            "avg_update_norm1": update_norm / num_params,
                            "avg_grad_norm1": grad_norm / num_params,
                        },
                        "debug/",
                    )
                    prev_params = curr_params

                    # gradient flow
                    ave_grads = []
                    max_grads = []
                    layers = []
                    for n, p in model.named_parameters():
                        if (p.requires_grad) and ("bias" not in n):
                            layers.append(n)
                            ave_grads.append(p.grad.abs().mean().item())
                            max_grads.append(p.grad.abs().max().item())

                    grad_flow = {
                        "layers": layers,
                        "ave_grads": ave_grads,
                        "max_grads": max_grads,
                    }
                    grad_flows.append(grad_flow)

                model.train()

        # Test model at end of batch
        with torch.no_grad():
            model.eval()
            test_mae = 0.0
            for data in dataloaders["test"]:
                data = {k: v.to(device) for k, v in data.items()}
                outputs = model(data, compute_loss=True)
                test_mae = test_mae + outputs.mae

        outputs["reports"].test_mae = test_mae / len(dataloaders["test"])

        reports = parse_reports(outputs.reports)

        log_tensorboard(summary_writer, train_iter, reports, "test")
        report_all = log_reports(report_all, train_iter, reports, "test")

        print_reports(
            reports,
            start_t,
            epoch,
            batch_idx,
            len(dataloaders["train"].dataset) // config.batch_size,
            prefix="test",
        )

        reports = {
            "lr": lr_schedule.get_lr()[0],
            "time": time.perf_counter() - start_t,
            "epoch": epoch,
        }

        log_tensorboard(summary_writer, train_iter, reports, "stats")
        report_all = log_reports(report_all, train_iter, reports, "stats")

        # Save the reports
        dd.io.save(logdir + "/results_dict.h5", report_all)

        # Save a checkpoint
        if epoch % config.save_check_points == 0:
            save_checkpoint(
                checkpoint_name, epoch, model, model_opt, lr_schedule, outputs.loss,
            )


if __name__ == "__main__":
    main()
