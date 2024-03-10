import torch

from eqv_transformer.eqv_attention import EquivariantTransformer
from eqv_transformer.mnist_predictor import MNISTClassifier
from lie_conv.lieGroups import SO3
from lie_conv.datasets import SO3aug

from forge import flags


flags.DEFINE_boolean(
    "data_augmentation",
    True,
    "Apply data augmentation to the data before passing to the model",
)
flags.DEFINE_integer("dim_hidden", 512, "Dimension of features to use in each layer")
flags.DEFINE_string(
    "activation_function", "swish", "Activation function to use in the network"
)
flags.DEFINE_boolean(
    "mean_pooling",
    True,
    "Use mean pooling insteave of sum pooling in the invariant layer",
)
flags.DEFINE_integer("num_heads", 8, "Number of attention heads in each layer")
flags.DEFINE_string(
    "block_norm",
    "layer_pre",
    "Type of norm to use in the attention block. none/[layer/batch]_[pre/post]",
)
flags.DEFINE_string(
    "output_norm",
    "none",
    "Type of norm to use in the final MLP layers block. none/layer/batch",
)
flags.DEFINE_string(
    "kernel_norm", "none", "The type of norm to use in the location kernels. none/batch"
)
flags.DEFINE_string(
    "kernel_type",
    "mlp",
    "Selects the type of attention kernel to use. mlp/relative_position/dot_product are valid",
)
flags.DEFINE_integer("kernel_dim", 16, "Hidden layer size to use in kernel MLPs")
flags.DEFINE_integer("num_layers", 6, "Number of ResNet layers to use")
flags.DEFINE_string("group", "SO3", "Group to be invariant to")
flags.DEFINE_integer(
    "lift_samples", 1, "Number of coset lift samples to use for non-trivial stabilisers"
)
flags.DEFINE_integer(
    "mc_samples",
    0,
    "Number of samples to use for estimating attention. 0 sets to use all points",
)
flags.DEFINE_float(
    "fill", 1.0, "Select mc_samples from K nearest mc_samples/fill points"
)
flags.DEFINE_integer("model_seed", 0, "Model rng seed")
flags.DEFINE_string(
    "architecture", "model_1", "The model architecture to use. model_1/lieconv"
)
flags.DEFINE_string(
    "attention_fn", "softmax", "Type of attention function to use. softmax/dot_product"
)
flags.DEFINE_integer(
    "feature_embed_dim",
    None,
    "Dimensionality of the embedding of the features for each head. Only used by some kernels",
)
flags.DEFINE_float(
    "max_sample_norm",
    None,
    "Maximum sample norm to allow through the lifting stage to prevent numerical issues.",
)
flags.DEFINE_string(
    "lie_algebra_nonlinearity",
    None,
    "Nonlinearity to apply to the norm of the lie algebra elements. Supported are None/tanh",
)
flags.DEFINE_boolean(
    "use_pseudo_lift", False, "Is valible, use the alternative pseudo lift method"
)
flags.DEFINE_boolean(
    "dual_quaternions",
    False,
    "If using SE3, chose if to use dual quaternion rep, or single quaternion plus 3 vec",
)
flags.DEFINE_boolean(
    "positive_quaternions",
    False,
    "If using quaternion rotations, limit the real part to be positive",
)


class MNISTEquivariantTransformer(EquivariantTransformer):
    def __init__(self, aug=False, group=SO3, **kwargs):
        super().__init__(dim_input=1, dim_output=10, group=group, **kwargs)
        self.aug = aug
        self.random_rotate = SO3aug()

    def forward(self, batch):
        x = batch
        with torch.no_grad():
            x = self.random_rotate(x) if self.aug else x
        return super().forward(x).squeeze(-1)
    
    def make_pvm(self, x):
        mask = ~torch.isnan(x[..., 0])
        p, v = x
        return (x, None, mask)


def load(config, **unused_kwargs):
    if config.group == "SO3":
        group = SO3(
            #0.2,
            use_pseudo=config.use_pseudo_lift,
            positive_quaternions=config.positive_quaternions,
        )
    else:
        raise ValueError(f"{config.group} is an invalid group")

    torch.manual_seed(config.model_seed)
    encoder = MNISTEquivariantTransformer(
        #config.num_species,
        #config.charge_scale,
        architecture=config.architecture,
        group=group,
        aug=config.data_augmentation,
        dim_hidden=config.dim_hidden,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        #global_pool=True, # default already True
        global_pool_mean=config.mean_pooling,
        liftsamples=config.lift_samples,
        block_norm=config.block_norm,
        output_norm=config.output_norm,
        kernel_norm=config.kernel_norm,
        kernel_type=config.kernel_type,
        kernel_dim=config.kernel_dim,
        kernel_act=config.activation_function,
        fill=config.fill,
        mc_samples=config.mc_samples,
        attention_fn=config.attention_fn,
        feature_embed_dim=config.feature_embed_dim,
        max_sample_norm=config.max_sample_norm,
        lie_algebra_nonlinearity=config.lie_algebra_nonlinearity,
    )

    mnist_predictor = MNISTClassifier(encoder)

    return (
        mnist_predictor,
        f"MNISTEquivariantTransformer_{config.group}_{config.architecture}",
    )
