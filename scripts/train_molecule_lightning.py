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
import pytorch_lightning as pl

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
    delete_checkpoint,
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
from oil.utils.utils import cosLr


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
    "lr_schedule",
    "none",
    "What learning rate schedule to use. Options: cosine, none",
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
flags.DEFINE_float(
    "lr_floor", 0, "minimum multiplicative factor of the learning rate in annealing"
)
flags.DEFINE_float(
    "warmup_length", 0.01, "fraction of the training time to use for warmup"
)
flags.DEFINE_bool(
    "find_spikes", False, "Find big spikes in validation loss and save checkpoints"
)
flags.DEFINE_boolean(
    "only_store_last_checkpoint",
    False,
    "If True, deletes last checkpoint when saving current checkpoint",
)
flags.DEFINE_boolean(
    "clip_grad",
    False,
    "If True, clip gradient L2-norms at 1.",
)

#####################################################################################################################

class MoleculeModule(pl.LightningModule):
    def __init__(self, model, config, summary_writer, checkpoint_name, n_train, n_valid):
        super().__init__()
        self.model = model
        self.config = config
        self.summary_writer = summary_writer
        self.checkpoint_name = checkpoint_name
        #self.dataloaders = dataloaders
        self.n_train = n_train
        self.n_valid = n_valid
        self.valid_mae = None
        self.outputs = None

        if self.config.clip_grad:
            raise NotImplementedError(
                "Gradient clipping not implemented for LightningModule"
            )
        if self.config.init_activations:
            raise NotImplementedError(
                "Activation initialisation not implemented for LightningModule"
            )
        if self.config.log_train_values:
            print("NOTENOTENOTE: log_train_values is set to True")
            #raise NotImplementedError("Logging train values not implemented for LightningModule")
        if self.config.debug:
            raise NotImplementedError("Debugging not implemented for LightningModule")
        if self.config.parameter_count:
            raise NotImplementedError("Parameter counting not implemented for LightningModule")
    
    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
    
    def configure_optimizers(self):
        model_params = self.model.predictor.parameters()
        opt_learning_rate = self.config.learning_rate

        model_opt = torch.optim.Adam(
            model_params,
            lr=opt_learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
        )

        # Cosine annealing learning rate
        if self.config.lr_schedule == "cosine":
            cos = cosLr(self.config.train_epochs)
            lr_sched = lambda e: max(cos(e), self.config.lr_floor * self.config.learning_rate)
            lr_schedule = optim.lr_scheduler.LambdaLR(model_opt, lr_sched)
        elif self.config.lr_schedule == "none":
            lr_sched = lambda e: 1.0
            lr_schedule = optim.lr_scheduler.LambdaLR(model_opt, lr_sched)
        else:
            raise ValueError(
                f"{self.config.lr_schedule} is not a recognised learning rate schedule"
            )
        
        return [model_opt], [lr_schedule]

    def training_step(self, batch, batch_idx):
        outputs = self(batch, compute_loss=True)
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, compute_loss=True)
        self.valid_mae += outputs.mae
        self.outputs = outputs
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return {k: v.to(device) for k, v in batch.items()}
        
    def on_fit_start(self):


        # Try to restore model and optimizer from checkpoint
        start_epoch = 1
        best_valid_mae = 1e12

        train_iter = (start_epoch - 1) * (
            self.n_train // self.config.batch_size
        ) + 1

        print("Starting training at epoch = {}, iter = {}".format(start_epoch, train_iter))


        report_all = defaultdict(list) 
        '''
        # Saving model at epoch 0 before training
        print("saving model at epoch 0 before training ... ")
        save_checkpoint(self.checkpoint_name, 0, self.model, model_opt, lr_schedule, 0.0)
        print("finished saving model at epoch 0 before training")
        '''

    def on_train_start(self):
        start_t = time.perf_counter()

        #iters_per_epoch = len(dataloaders["train"])
        last_valid_loss = 1000.0

    def on_validation_epoch_start(self):
        self.valid_mae = 0.0
    
    def on_validation_epoch_end(self):
        outputs = self.outputs

        self.valid_mae /= self.n_valid
        outputs["reports"].valid_mae = self.valid_mae
        reports = parse_reports(self.outputs.reports)

        # TODO: all this logging and checkpointing should somewhere else...
        '''
        log_tensorboard(self.summary_writer, train_iter, reports, "valid")
        report_all = log_reports(report_all, train_iter, reports, "valid")
        print_reports(
            reports,
            start_t,
            epoch,
            batch_idx,
            self.n_train // self.config.batch_size,
            prefix="valid",
        )

        loss_diff = (
            last_valid_loss - (self.valid_mae / len(self.dataloaders["valid"])).item()
        )
        if loss_diff and self.config.find_spikes < -0.1:
            save_checkpoint(
                checkpoint_name + "_spike",
                epoch,
                self.model,
                model_opt,
                lr_schedule,
                outputs.loss,
            )

        last_valid_loss = (self.valid_mae / len(self.dataloaders["valid"])).item()

        if outputs["reports"].valid_mae < best_valid_mae:
            save_checkpoint(
                checkpoint_name,
                "best_valid_mae",
                self.model,
                model_opt,
                lr_schedule,
                best_valid_mae,
            )
            best_valid_mae = outputs["reports"].valid_mae
        '''
'''
    def on_train_epoch_end(self):
        # Save a checkpoint
        if self.current_epoch % self.config.save_check_points == 0:
            save_checkpoint(
                self.checkpoint_name,
                self.current_epoch,
                self.model,
                model_opt,
                lr_schedule,
                best_valid_mae,
            )
            if self.config.only_store_last_checkpoint:
                delete_checkpoint(self.checkpoint_name, self.current_epoch - self.config.save_check_points)

    def on_fit_end(self):
        save_checkpoint(
            self.checkpoint_name,
            "final",
            self.model,
            model_opt,
            lr_schedule,
            self.outputs.loss,
        )
'''



# @forge.debug_on(Exception)
def main():
    # Parse flags
    config = forge.config()
    #print(config.__dict__['__flags'])

    # Load data
    dataloaders, num_species, charge_scale, ds_stats, data_name = fet.load(
        config.data_config, config=config
    )

    config.num_species = num_species
    config.charge_scale = charge_scale
    config.ds_stats = ds_stats

    # Load model
    model, model_name = fet.load(config.model_config, config)

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

    # Print flags
    fet.print_flags()


    # set up results folders
    results_folder_name = osp.join(
        data_name,
        model_name,
        run_name,
    )

    logdir = osp.join(config.results_dir, results_folder_name.replace(".", "_"))
    logdir, resume_checkpoint = fet.init_checkpoint(
        logdir, config.data_config, config.model_config, config.resume
    )

    checkpoint_name = osp.join(logdir, "model.ckpt")

    # Setup tensorboard writing
    summary_writer = SummaryWriter(logdir)

    num_params = param_count(model)
    print(f"{model_name} parameters: {num_params:.5e}")

    n_train = len(dataloaders["train"].dataset)
    n_valid = len(dataloaders["valid"].dataset)

    model = MoleculeModule(model, config, summary_writer, checkpoint_name, n_train, n_valid)
    trainer = pl.Trainer(default_root_dir=logdir, max_epochs=config.train_epochs)

    trainer.fit(model=model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["valid"])
    #trainer.test(model=model)
    '''
    #for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):

        #for batch_idx, data in enumerate(dataloaders["train"]):

            #outputs = model(data, compute_loss=True)

            #outputs.loss.backward()
            #model_opt.step()

            # Logging

            train_iter += 1

            # Step the LR schedule
            lr_schedule.step(train_iter / iters_per_epoch)

        # Test model at end of batch
        #with torch.no_grad():
        #    model.eval()
        #    test_mae = 0.0
        #    for data in dataloaders["test"]:
        #        data = {k: v.to(device) for k, v in data.items()}
        #        outputs = model(data, compute_loss=True)
        #        test_mae = test_mae + outputs.mae

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
                checkpoint_name,
                epoch,
                model,
                model_opt,
                lr_schedule,
                best_valid_mae,
            )
            if config.only_store_last_checkpoint:
                delete_checkpoint(checkpoint_name, epoch - config.save_check_points)
    '''

if __name__ == "__main__":
    main()
