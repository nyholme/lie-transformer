
import os

import torch
from torchvision import datasets, transforms

from forge import flags


flags.DEFINE_string('data_folder', 'data/MNIST_data', 'Path to data folder.')


def load(config, **unused_kwargs):
    del unused_kwargs

    if not os.path.exists(config.data_folder):
        os.makedirs(config.data_folder)

    mnist = datasets.MNIST(config.data_folder, train=True, download=True,
                           transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        mnist, batch_size=config.batch_size, shuffle=True)

    return train_loader
