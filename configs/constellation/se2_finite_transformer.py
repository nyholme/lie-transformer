from attrdict import AttrDict

import torch
from torch import nn
import torch.nn.functional as F
from eqv_transformer.classfier import Classifier
from eqv_transformer.eqv_attention_se2_finite import EqvTransformer

from forge import flags

# flags.DEFINE_integer('input_dim', 2, 'Dimensionality of the input.')
flags.DEFINE_integer("n_outputs", 4, "Number of output vectors.")
# flags.DEFINE_integer('output_dim', 3, 'Dimensionality of the output.')
flags.DEFINE_string("content_type", "pairwise_distances", "How to initialize y")
flags.DEFINE_integer("n_enc_layers", 4, "Number of encoder layers.")
flags.DEFINE_integer("n_dec_layers", 4, "Number of encoder layers.")
flags.DEFINE_integer("n_heads", 4, "Number of attention heads.")
flags.DEFINE_boolean("layer_norm", False, "Uses layer-norm if True.")
flags.DEFINE_integer("cn", 5, "Size of rotation group.")


def load(config, **unused_kwargs):
    del unused_kwargs

    input_dim = config.patterns_reps * 17 - 1
    output_dim = config.patterns_reps + 1

    encoder = EqvTransformer(
        input_dim,
        config.n_outputs,
        output_dim,
        config.content_type,
        n_enc_layers=config.n_enc_layers,
        n_dec_layers=config.n_dec_layers,
        num_heads=config.n_heads,
        ln=config.layer_norm,
        cn=config.cn,
    )

    classifier = Classifier(encoder)
    return classifier, "SE2_EquivariantTransformer"
