# flake8: noqa
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Union

import tensorflow as tf
from keras import Model
from keras.layers import LSTM as LSTMLayer
from keras.layers import Dense, TimeDistributed
from keras.models import load_model
from keras.optimizers import Adam
from keras.src.engine import data_adapter

from .config import LSTMModelConfig
from .losses import weighted_loss

Model.train_step

Model.build_from_config


class LSTM(Model):
    def build_from_config(self, config: Dict) -> None:
        input_shape = config["input_shape"]
        assert len(input_shape) == 2, input_shape
        encoder_input_shape, decoder_input_shape = input_shape
        if input_shape is not None:
            self.build(encoder_input_shape)

    def test_step(
        self, data: Tuple[tf.Tensor, tf.Tensor]
    ) -> Dict[str, float]:
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        x, _ = x
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

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

    def __init__(self, model_config: LSTMModelConfig, **kwargs):
        # Define model parameters
        self.model_config = model_config
        super().__init__(**kwargs)

        # Define optimizer
        self.optimizer = Adam(learning_rate=model_config.lr)

        # Encoder
        ann = load_model(model_config.ann_model_path)
        assert isinstance(ann, Model), type(ann)
        ann.trainable = False
        for layer in ann.layers:
            layer.trainable = False

        # Propagate this input through ann
        input_tensor = ann.layers[0].output
        x = input_tensor
        # start from the layer after the InputLayer and skip the last layer
        for layer in ann.layers[1:-1]:
            x = layer(x)
        self.encoder = Model(inputs=input_tensor, outputs=x)
        self.encoder.trainable = False
        self.state_h_transform = Dense(
            ann.model_config.n2,
            activation=model_config.state_transform_activation,
            input_shape=(ann.layers[-2].units,),
        )
        self.state_c_transform = Dense(
            ann.model_config.n2,
            activation=model_config.state_transform_activation,
            input_shape=(ann.layers[-2].units,),
        )

        # Decoder
        self.decoder_lstm = LSTMLayer(
            ann.model_config.n2,
            return_sequences=True,
            return_state=True,
            input_shape=(model_config.seq_len, model_config.dim_out),
        )
        self.decoder_dense = TimeDistributed(Dense(model_config.dim_out))
        self.compile(
            optimizer=self.optimizer,
            loss=model_config.loss_funcs,
            metrics=model_config.metrics,
        )

    def call(
        self,
        inputs: Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor],
        training=False,
    ):
        if training:
            encoder_input, decoder_input = inputs
            # encoder_output = self.encoder(encoder_input)
            # state_h = self.state_h_transform(encoder_output)
            # state_c = self.state_c_transform(encoder_output)
            # decoder_output, _, _ = self.decoder_lstm(  # type: ignore
            #     decoder_input, initial_state=[state_h, state_c]
            # )
            # return self.decoder_dense(decoder_output)
        else:
            encoder_input = inputs
        encoder_output = self.encoder(encoder_input)
        state_h = self.state_h_transform(encoder_output)
        state_c = self.state_c_transform(encoder_output)
        decoder_buffer = tf.zeros(
            (tf.shape(encoder_output)[0], 1, self.model_config.dim_out)
        )

        initial_states = (decoder_buffer, state_h, state_c)

        def step_fn(previous_states, _):
            decoder_buffer, state_h, state_c = previous_states
            lstm_out, state_h, state_c = self.decoder_lstm(  # type: ignore
                decoder_buffer, initial_state=[state_h, state_c]
            )
            decoder_buffer = self.decoder_dense(lstm_out)
            return (decoder_buffer, state_h, state_c)

        all_outputs = tf.scan(
            fn=step_fn,
            elems=tf.range(self.model_config.seq_len),
            initializer=initial_states,
        )[0]

        return tf.reshape(
            all_outputs,
            [-1, self.model_config.seq_len, self.model_config.dim_out],
        )


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

# from .config import LSTMModelConfig


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


# class LSTMFrame(Model):
#     def train_step(
#         self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
#     ) -> Dict[str, Union[float, tf.Tensor]]:
#         (
#             x,
#             y,
#             sample_weight,
#         ) = data_adapter.unpack_x_y_sample_weight(data)

#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             tf.print(
#                 "y_pred:",
#                 y_pred[0, 0, 0],  # type: ignore
#                 y_pred[0, tf.shape(y_pred)[1] // 2, 0],  # type: ignore
#                 y_pred[0, -1, 0],  # type: ignore
#             )
#             tf.print(
#                 "y_true:",
#                 y[0, 0, 0],  # type: ignore
#                 y[0, tf.shape(y)[1] // 2, 0],  # type: ignore
#                 y[0, -1, 0],  # type: ignore
#             )
#             loss = self.compiled_loss(y, y_pred)  # type: ignore

#         # Compute gradients and update weights
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         # Update and return metrics (assuming you've compiled the model with some metrics)
#         if self.compiled_metrics is not None:
#             self.compiled_metrics.update_state(y, y_pred, sample_weight)
#             return {m.name: m.result() for m in self.metrics}
#         else:
#             return {}
# def train_step(self, data):
#     """The logic for one training step.
#     This method is called by `Model.make_train_function`.

#     Args:
#       data: A nested structure of `Tensor`s.
#     Returns:
#       A `dict` containing values that will be passed to
#       `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
#       values of the `Model`'s metrics are returned. Example:
#       `{'loss': 0.2, 'accuracy': 0.7}`.
#     """
#     x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
#     # Run forward pass.
#     with tf.GradientTape() as tape:
#         y_pred = self(x, training=True)
#         loss = self.compute_loss(x, y, y_pred, sample_weight)
#     self._validate_target_and_loss(y, loss)
#     # Run backwards pass.
#     self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
#     return self.compute_metrics(x, y, y_pred, sample_weight)
