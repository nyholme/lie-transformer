
from torch import nn
from attrdict import AttrDict

import torch
import torch.nn.functional as F


class MNISTPredictor(nn.Module):
    """This class implements forward pass through our model, including loss computation."""

    def __init__(self, predictor, ds_stats=None):
        """Builds the model.

        Args:
            predictor: callable that takes an input, e.g. a molecule, and returns a prediction of a property of the molecule.
            ds_stats: normalisation mean and variance of the targets. If None, do no normalisation.
        Returns:
            a dictionary of outputs.
        """
        super().__init__()
        self.predictor = predictor
        self.ds_stats = ds_stats

    def forward(self, inpt):

        o = AttrDict()

        o.prediction = self.predictor(inpt)

        # label = label.long()
        target = inpt[self.task]

        # import pdb;
        # pdb.set_trace()
        if self.ds_stats is not None:
            meadian, mad = self.ds_stats

        if self.ds_stats is not None:
            meadian, mad = self.ds_stats

            target_norm = (target - meadian) / mad
            prediction_actual = o.prediction * mad + meadian
            o.prediction_actual = prediction_actual

            o.loss = (o.prediction - target_norm).abs().mean()
            o.mae = (prediction_actual - target).abs().mean()
        else:
            o.loss = (o.prediction - target).abs().mean()
            o.mae = o.loss
        # import pdb;
        # pdb.set_trace()

        o.reports = AttrDict({"loss": o.loss, "mae": o.mae})

        return o


class MNISTClassifier(nn.Module):
    """This class implements forward pass through our model, including loss computation."""

    def __init__(self, encoder):
        """Builds the model.

        Args:
            encoder: callable that takes an input, e.g. an image, and returns a representation.
        Returns:
            a dictionary of outputs.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, batch):

        o = AttrDict()

        # compute model forward pass
        #if isinstance(inpt, (list, tuple)):
        #    o.logits = self.encoder(*inpt)
        #else:
        p, v, m, target = batch
        data = (p, v, m)
        o.logits = self.encoder(data)

        # compute the predicted classes
        _, o.predicted = torch.max(o.logits, -1)

        #*_, target = batch
        #label = label.long()

        o.loss = F.cross_entropy(o.logits.transpose(1, 2), target)
        o.acc = o.predicted.eq(target).float().mean(())

        o.reports = AttrDict({"loss": o.loss, "acc": o.acc})

        return o