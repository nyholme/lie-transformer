import copy
import time
import torch
from torch import nn

import forge.experiment_tools as fet


def parse_reports(report_dict):
    return {k: v.item() if len(v.shape) == 0 else v.detach().clone() for k, v in report_dict.items()}


def print_reports(report_dict, start_time, epoch, batch_idx, num_epochs, prefix=''):

    reports = ['{}:{:.03f}'.format(*item) for item in report_dict.items()]
    report_string = ', '.join(reports)
    if prefix:
        print(prefix, end=': ')

    print('time {:.03f},  epoch: {} [{} / {}]: {}'.format(
        time.perf_counter() - start_time, epoch, batch_idx, num_epochs, report_string))


def log_tensorboard(writer, iteration, report_dict, prefix=''):
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'

    for k, v in report_dict.items():
        writer.add_scalar(prefix + k, v, iteration)


def get_checkpoint_iter(checkpoint_iter, checkpoint_dir):
    if checkpoint_iter != -1:
        return checkpoint_iter

    return max(fet.find_model_files(checkpoint_dir).keys())


def load_checkpoint(checkpoint_path, model, opt):
    print("Restoring checkpoint from '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    # Restore model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer
    opt.load_state_dict(checkpoint['model_optimizer_state_dict'])
    # Update starting epoch
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def save_checkpoint(checkpoint_name, epoch, model, opt, loss=None):
    epoch_ckpt_file = '{}-{}'.format(checkpoint_name, epoch)
    print("Saving model training checkpoint to {}".format(epoch_ckpt_file))

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_optimizer_state_dict': opt.state_dict(),
    }

    if loss is not None:
        state['loss'] = loss

    torch.save(state, epoch_ckpt_file)
    return epoch_ckpt_file


class ExponentialMovingAverage(nn.Module):

    def __init__(self, alpha=.99, initial_value=0., debias=False):
        super(ExponentialMovingAverage, self).__init__()

        self.alpha = alpha
        self.initial_value = initial_value
        self.debias = debias
        if self.debias and self.initial_value != 0:
            raise NotImplementedError('Debiasing is implemented only for initial_value==0.')

        self.ema = None
        self.alpha_power = 1.

    def forward(self, x):
        """x can be a scalar, a tensor, or a dict of scalars or tensors."""

        if self.ema is None:
            if isinstance(x, dict):
                self.ema = x.__class__({k: self.initial_value for k in x})
            else:
                self.ema = self.initial_value

        am1 = 1. - self.alpha
        if isinstance(x, dict):
            for k, v in x.items():
                self.ema[k] = self.ema[k] * self.alpha + v * am1
            ema = copy.deepcopy(self.ema)
        else:
            self.ema = self.ema * self.alpha + x * am1
            ema = self.ema

        if self.debias and self.alpha_power > 0.:
            self.alpha_power *= self.alpha
            apm1 = 1. - self.alpha_power
            
            if isinstance(ema, dict):
                for k in ema:
                    ema[k] /= apm1
            else:
                ema /= apm1

        return ema