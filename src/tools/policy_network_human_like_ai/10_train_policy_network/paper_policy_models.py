"""CNN architectures from Liskowski et al., IEEE TG 2018, Tables I and V.

DOI: 10.1109/TG.2018.2799997. The paper specifies the architecture but not
the exact BN placement/hyperparameters. Here BN precedes ReLU in each
convolution and the hidden dense layer (momentum=0.99, epsilon=1e-3).
Training retains this project's optimizer, data split, and evaluation rules.
"""

from tensorflow.keras.layers import (
    BatchNormalization, Concatenate, Conv2D, Cropping1D, Dense, Flatten,
    Input, Permute, ReLU, Reshape, Softmax, ZeroPadding1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 as keras_l2


PAPER_FILTERS = {
    "conv4": (64, 64, 128, 128),
    "conv6": (64, 64, 128, 128, 256, 256),
    "conv8": (64, 64, 128, 128, 256, 256, 256, 256),
    "conv8_bn": (64, 64, 128, 128, 256, 256, 256, 256),
}
# The initial four discs can never be played on again. Policy indices are
# ascending bitboard indices, matching the existing 64-way sparse labels.
CENTER_POLICIES = (27, 28, 35, 36)
PAPER_POLICIES = tuple(p for p in range(64) if p not in CENTER_POLICIES)


def expand_paper_policy(probabilities):
    """Scatter 60 softmax probabilities to 64 squares without trainable weights.

    Only built-in Keras layers are used, so .h5 files load without custom_objects.
    The three contiguous runs are 0..26, 29..34, and 37..63.
    """
    column = Reshape((60, 1), name="policy60_column")(probabilities)
    first = Cropping1D((0, 33), name="policy_first27")(column)
    middle = Cropping1D((27, 27), name="policy_middle6")(column)
    last = Cropping1D((33, 0), name="policy_last27")(column)
    first = ZeroPadding1D((0, 2), name="policy_zero_d5_e5")(first)
    middle = ZeroPadding1D((0, 2), name="policy_zero_d4_e4")(middle)
    joined = Concatenate(axis=1, name="policy64_column")([first, middle, last])
    return Flatten(name="policy")(joined)


def build_paper_model(architecture: str, name: str, l2_value: float) -> Model:
    filters = PAPER_FILTERS[architecture]
    batch_norm = architecture == "conv8_bn"
    regularizer = keras_l2(l2_value) if l2_value > 0.0 else None
    inputs = Input(shape=(128,), name="board")
    # Existing features contain the whole player plane, then the opponent plane.
    x = Reshape((2, 8, 8), name="pieces_planes")(inputs)
    x = Permute((2, 3, 1), name="pieces_channels_last")(x)
    for i, channels in enumerate(filters):
        x = Conv2D(
            channels, 3, strides=1, padding="same", data_format="channels_last",
            use_bias=not batch_norm, kernel_initializer="he_normal",
            bias_initializer="zeros", kernel_regularizer=regularizer,
            name=f"conv_{i}",
        )(x)
        if batch_norm:
            x = BatchNormalization(momentum=0.99, epsilon=1e-3, name=f"conv_bn_{i}")(x)
        x = ReLU(name=f"conv_relu_{i}")(x)
    x = Flatten(name="conv_flatten")(x)
    x = Dense(
        128, use_bias=not batch_norm, kernel_initializer="he_normal",
        bias_initializer="zeros", kernel_regularizer=regularizer, name="fc128",
    )(x)
    if batch_norm:
        x = BatchNormalization(momentum=0.99, epsilon=1e-3, name="fc128_bn")(x)
    x = ReLU(name="fc128_relu")(x)
    logits = Dense(
        60, kernel_initializer="he_normal", bias_initializer="zeros",
        kernel_regularizer=regularizer, name="policy_logits60",
    )(x)
    probabilities = Softmax(name="policy60")(logits)
    return Model(inputs, expand_paper_policy(probabilities), name=name)
