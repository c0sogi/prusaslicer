# flake8: noqa
from dataclasses import asdict
from typing import Dict, Union

import tensorflow as tf
from keras import Model
from keras.layers import LSTM as LSTMLayer
from keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    RepeatVector,
    TimeDistributed,
)
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.src.engine import data_adapter

from .config import LSTMModelConfig
from .losses import weighted_loss


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
            model_config=LSTMModelConfig.from_dict(
                config.pop("model_config")
            ),
            **config,
        )


class LSTM(ModelFrame):
    def __init__(self, model_config: LSTMModelConfig, **kwargs):
        # Define model parameters
        self.model_config = model_config
        super().__init__(**kwargs)

        # Encoder
        self.encoder_dense = Dense(
            model_config.encoder_lstm_units, activation="relu"
        )
        self.encoder_lstm = LSTMLayer(
            model_config.encoder_lstm_units, return_state=True
        )

        # Decoder
        self.repeat_vector = RepeatVector(model_config.max_sequence_length)
        self.decoder_lstm = LSTMLayer(
            model_config.decoder_lstm_units, return_sequences=True
        )
        self.decoder_dense = TimeDistributed(Dense(model_config.dim_out))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Encoder
        x = tf.expand_dims(
            self.encoder_dense(inputs), 1
        )  # Add the time dimension
        e_out = self.encoder_lstm(x)
        assert e_out is not None, "Encoder LSTM returns None"
        encoder_outputs, state_h, state_c = e_out
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = self.repeat_vector(encoder_outputs)
        decoder_outputs = self.decoder_dense(
            self.decoder_lstm(decoder_inputs, initial_state=encoder_states)
        )
        assert decoder_outputs is not None, "Decoder Dense returns None"
        return decoder_outputs


# import tensorflow as tf
# from keras.layers import (
#     LSTM,
#     Dense,
#     Dropout,
#     LayerNormalization,
#     TimeDistributed,
# )
# from keras.models import Model
# from keras.optimizers import Adam
# from tensorflow.python.keras.engine import data_adapter

# from nn.config import LSTMModelConfig


# class LSTMModel(ModelFrame):
#     def __init__(self, model_config: LSTMModelConfig, **kwargs):
#         self.model_config = model_config
#         super().__init__(**kwargs)

#         # Define optimizer
#         self.optimizer = Adam(learning_rate=model_config.lr)

#         # Define regularization
#         if model_config.l1_reg is None or model_config.l2_reg is None:
#             kernel_regularizer = None
#         else:
#             kernel_regularizer = l1_l2(
#                 l1=model_config.l1_reg, l2=model_config.l2_reg
#             )

#         # Define layers
#         activation = model_config.activation

#         self.lstm1 = LSTM(
#             units=model_config.n1,
#             return_sequences=True,
#             activation=activation,
#             kernel_regularizer=kernel_regularizer,
#         )
#         self.norm1 = (
#             LayerNormalization() if model_config.normalize_layer else None
#         )
#         self.dropout1 = (
#             Dropout(model_config.dropout_rate)
#             if model_config.dropout_rate
#             else None
#         )

#         self.lstm2 = LSTM(
#             units=model_config.n2,
#             return_sequences=True,
#             activation=activation,
#             kernel_regularizer=kernel_regularizer,
#         )
#         self.norm2 = (
#             LayerNormalization() if model_config.normalize_layer else None
#         )
#         self.dropout2 = (
#             Dropout(model_config.dropout_rate)
#             if model_config.dropout_rate
#             else None
#         )

#         self.lstm3 = LSTM(
#             units=model_config.n3,
#             return_sequences=True,
#             activation=activation,
#             kernel_regularizer=kernel_regularizer,
#         )
#         self.norm3 = (
#             LayerNormalization() if model_config.normalize_layer else None
#         )
#         self.dropout3 = (
#             Dropout(model_config.dropout_rate)
#             if model_config.dropout_rate
#             else None
#         )

#         self.time_distributed_out = TimeDistributed(
#             Dense(units=model_config.dim_out)
#         )

#         self.compile(
#             optimizer=self.optimizer,
#             loss=weighted_loss(
#                 *model_config.loss_weights,
#                 loss_funcs=model_config.loss_funcs,
#             ),
#             metrics=model_config.metrics,
#         )

#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         x = self.lstm1(inputs)
#         if self.norm1:
#             x = self.norm1(x)
#         if self.dropout1:
#             x = self.dropout1(x)
#         x = self.lstm2(x)
#         if self.norm2:
#             x = self.norm2(x)
#         if self.dropout2:
#             x = self.dropout2(x)
#         x = self.lstm3(x)
#         if self.norm3:
#             x = self.norm3(x)
#         if self.dropout3:
#             x = self.dropout3(x)
#         return self.time_distributed_out(x)
