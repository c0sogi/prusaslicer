# flake8: noqa
from dataclasses import asdict
from typing import Dict, Union

import tensorflow as tf
from keras import Model, initializers
from keras.layers import Dense, Dropout, InputLayer, LayerNormalization
from keras.optimizers import Adam
from keras.regularizers import l1, l1_l2, l2
from keras.src.engine import data_adapter

from .config import ANNModelConfig
from .losses import weighted_loss


class ANN(Model):
    def get_config(self):
        config = super().get_config()
        config.update({"model_config": asdict(self.model_config)})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            model_config=ANNModelConfig.from_dict(
                config.pop("model_config")
            ),
            **config,
        )

    def __init__(self, model_config: ANNModelConfig, **kwargs):
        # Define model parameters
        self.model_config = model_config
        super().__init__(**kwargs)

        # Define optimizer
        self.optimizer = Adam(learning_rate=model_config.lr)

        # Define regularization
        if model_config.l1_reg and model_config.l2_reg:
            kernel_regularizer = l1_l2(
                l1=model_config.l1_reg, l2=model_config.l2_reg
            )
        elif model_config.l1_reg:
            kernel_regularizer = l1(model_config.l1_reg)
        elif model_config.l2_reg:
            kernel_regularizer = l2(model_config.l2_reg)
        else:
            kernel_regularizer = None

        # Define layers
        activation = model_config.activation
        self.input_layer = InputLayer(input_shape=(model_config.dim_in,))
        self.dense1 = Dense(
            units=model_config.n1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.norm1 = (
            LayerNormalization() if model_config.normalize_layer else None
        )
        self.dropout1 = (
            Dropout(model_config.dropout_rate)
            if model_config.dropout_rate
            else None
        )

        self.dense2 = Dense(
            units=model_config.n2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.norm2 = (
            LayerNormalization() if model_config.normalize_layer else None
        )
        self.dropout2 = (
            Dropout(model_config.dropout_rate)
            if model_config.dropout_rate
            else None
        )
        self.dense3 = Dense(
            units=model_config.n3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.norm3 = (
            LayerNormalization() if model_config.normalize_layer else None
        )
        self.dropout3 = (
            Dropout(model_config.dropout_rate)
            if model_config.dropout_rate
            else None
        )
        self.dense_out = Dense(units=model_config.dim_out)
        self.compile(
            optimizer=self.optimizer,
            loss=weighted_loss(
                *model_config.loss_weights,
                loss_funcs=model_config.loss_funcs,
            ),
            metrics=model_config.metrics,
        )

    def call(self, inputs: tf.Tensor, training=False):
        x = self.dense1(self.input_layer(inputs))

        if self.norm1 is not None:
            x = self.norm1(x, training=training)
        if self.dropout1 is not None:
            x = self.dropout1(x, training=training)

        x = self.dense2(x)
        if self.norm2 is not None:
            x = self.norm2(x, training=training)
        if self.dropout2 is not None:
            x = self.dropout2(x, training=training)

        x = self.dense3(x)
        if self.norm3 is not None:
            x = self.norm3(x, training=training)
        if self.dropout3 is not None:
            x = self.dropout3(x, training=training)

        return self.dense_out(x)
