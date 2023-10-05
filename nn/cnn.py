# flake8: noqa
from dataclasses import asdict
from typing import Callable, Dict, List, Literal, Optional, Union

import tensorflow as tf
from keras import Model, Sequential, initializers, losses
from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    Layer,
    LayerNormalization,
    MaxPooling1D,
)
from keras.optimizers import Adam

from keras.regularizers import l1_l2
from keras.src.engine import data_adapter

from .config import BaseModelConfig, CNNModelConfig
from .losses import weighted_loss

PoolingTypes = Literal[
    "max", "average", "global_max", "global_average", "none"
]


class ModelFrame(Model):
    def train_step(
        self, data: tf.Tensor
    ) -> Dict[str, Union[float, tf.Tensor]]:
        (
            x,
            y,
            sample_weight,
        ) = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred)  # type: ignore

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update and return metrics (assuming you've compiled the model with some metrics)
        if self.compiled_metrics is not None:
            self.compiled_metrics.update_state(y, y_pred, sample_weight)
            return {m.name: m.result() for m in self.metrics}
        else:
            return {}

    def get_config(self):
        config = super().get_config()
        config.update({"model_config": asdict(self.model_config)})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            model_config=CNNModelConfig.from_dict(
                config.pop("model_config")
            ),
            **config,
        )


class CNN(ModelFrame):
    def __init__(self, model_config: BaseModelConfig, **kwargs):
        super().__init__(**kwargs)

        # Define optimizer
        self.optimizer = Adam(learning_rate=model_config.lr)

        # Define regularization
        if model_config.l1_reg is None or model_config.l2_reg is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = l1_l2(
                l1=model_config.l1_reg, l2=model_config.l2_reg
            )

        # Define layers
        activation = model_config.activation

        self.conv1 = Conv1D(
            filters=model_config.n1,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.pool1 = MaxPooling1D(pool_size=2)
        self.norm1 = (
            LayerNormalization() if model_config.normalize_layer else None
        )
        self.dropout1 = (
            Dropout(model_config.dropout_rate)
            if model_config.dropout_rate
            else None
        )

        self.conv2 = Conv1D(
            filters=model_config.n2,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.pool2 = MaxPooling1D(pool_size=2)
        self.norm2 = (
            LayerNormalization() if model_config.normalize_layer else None
        )
        self.dropout2 = (
            Dropout(model_config.dropout_rate)
            if model_config.dropout_rate
            else None
        )

        self.flatten = Flatten()

        self.dense_out = Dense(units=model_config.dim_out)

        self.compile(
            optimizer=self.optimizer,
            loss=weighted_loss(
                *model_config.loss_weights,
                loss_funcs=model_config.loss_funcs,
            ),
            metrics=model_config.metrics,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.pool1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)

        x = self.flatten(x)

        return self.dense_out(x)  # type: ignore
