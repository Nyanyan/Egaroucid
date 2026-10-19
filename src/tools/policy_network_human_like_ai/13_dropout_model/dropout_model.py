"""Attach inference-time dropout to an existing WTHOR network without training."""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "10_train_policy_network"))

from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                                     Flatten, Input, Permute, ReLU, Reshape, Softmax)
from tensorflow.keras.models import Model
from paper_policy_models import PAPER_FILTERS, expand_paper_policy


def validate_dropout_rate(value: float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("dropout rate must be finite and satisfy 0 <= rate < 1")
    return rate


def build_dropout_model(dropout_rate: float = 0.2) -> Model:
    """Apply ordinary element-wise Dropout after all nine hidden ReLU layers.

    A rate of zero reproduces the original Conv8+BN graph. Only built-in Keras
    layers are used so saved H5 files need no custom_objects at inference time.
    """
    rate = validate_dropout_rate(dropout_rate)
    inputs = Input(shape=(128,), name="board")
    x = Reshape((2, 8, 8), name="pieces_planes")(inputs)
    x = Permute((2, 3, 1), name="pieces_channels_last")(x)
    for index, channels in enumerate(PAPER_FILTERS["conv8_bn"]):
        x = Conv2D(channels, 3, strides=1, padding="same", data_format="channels_last",
                   use_bias=False, kernel_initializer="he_normal", name=f"conv_{index}")(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3, name=f"conv_bn_{index}")(x)
        x = ReLU(name=f"conv_relu_{index}")(x)
        if rate:
            x = Dropout(rate, name=f"conv_dropout_{index}")(x)
    x = Flatten(name="conv_flatten")(x)
    x = Dense(128, use_bias=False, kernel_initializer="he_normal", name="fc128")(x)
    x = BatchNormalization(momentum=0.99, epsilon=1e-3, name="fc128_bn")(x)
    x = ReLU(name="fc128_relu")(x)
    if rate:
        x = Dropout(rate, name="fc128_dropout")(x)
    logits = Dense(60, kernel_initializer="he_normal", bias_initializer="zeros",
                   dtype="float32", name="policy_logits60")(x)
    probabilities = Softmax(name="policy60", dtype="float32")(logits)
    return Model(inputs, expand_paper_policy(probabilities), name="conv8_bn_dropout")


def build_inference_dropout_model(pretrained, dropout_rate=0.2, seed=613):
    """Clone fixed weights into float32, freeze BN, and force only Dropout on.

    Each hidden ReLU is followed by a standard Dropout layer. The Dropout
    call stores training=True in the graph, so model.predict activates it while
    every original layer (including BN) remains in inference mode. H5 files
    need no custom_objects. No fit, compile, optimizer, or weight update occurs.
    """
    import numpy as np
    import tensorflow as tf
    rate = validate_dropout_rate(dropout_rate)
    names = [f"conv_relu_{index}" for index in range(8)] + ["fc128_relu"]
    if any(not isinstance(pretrained.get_layer(name), ReLU) for name in names):
        raise ValueError("Expected the pretrained Conv8+BN hidden ReLU layers")

    def apply(name, values):
        layer = pretrained.get_layer(name)
        config = layer.get_config()
        config["dtype"] = "float32"
        copied = layer.__class__.from_config(config)
        return copied(values)

    inputs = Input(shape=(128,), dtype="float32", name="board")
    x = apply("pieces_planes", inputs)
    x = apply("pieces_channels_last", x)
    for index in range(8):
        x = apply(f"conv_{index}", x)
        x = apply(f"conv_bn_{index}", x)
        x = apply(f"conv_relu_{index}", x)
        if rate:
            x = Dropout(rate, seed=seed+index, dtype="float32",
                        name=f"conv_dropout_{index}")(x, training=True)
    x = apply("conv_flatten", x)
    x = apply("fc128", x)
    x = apply("fc128_bn", x)
    x = apply("fc128_relu", x)
    if rate:
        x = Dropout(rate, seed=seed+8, dtype="float32",
                    name="fc128_dropout")(x, training=True)
    logits = apply("policy_logits60", x)
    probabilities = apply("policy60", logits)
    model = Model(inputs, expand_paper_policy(probabilities), name="wthor_inference_dropout")
    model.set_weights(pretrained.get_weights())
    model.trainable = False
    if model.count_params() != pretrained.count_params():
        raise ValueError("Dropout must not change the learned parameter count")
    if any(not np.array_equal(a, b) for a, b in zip(model.get_weights(), pretrained.get_weights())):
        raise ValueError("Pretrained weights or BN statistics changed")
    return model
