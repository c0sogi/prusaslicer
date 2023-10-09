# flake8: noqa
from dataclasses import asdict
from typing import Dict, List, Union

import tensorflow as tf
from keras import Model
from keras.layers import LSTM as LSTMLayer
from keras.layers import Dense, TimeDistributed
from keras.models import load_model
from keras.optimizers import Adam
from keras.src.engine import data_adapter

from .config import LSTMModelConfig
from .losses import weighted_loss


class LSTMFrame(Model):
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


class LSTM(LSTMFrame):
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

        # Propagate this input through all layers of 'ann' up to 'dense_2'
        input_tensor = ann.layers[0].output
        x = input_tensor
        for layer in ann.layers[1:-1]:
            # start from the layer after the InputLayer and skip the last layer
            x = layer(x)

        # Now x holds the output of 'dense_2'. We create a new model with input_tensor as input and x as output.
        self.encoder = Model(inputs=input_tensor, outputs=x)
        self.encoder.trainable = False
        self.state_h_transform = Dense(
            ann.model_config.n2,
            activation=model_config.state_transform_activation,
        )
        self.state_c_transform = Dense(
            ann.model_config.n2,
            activation=model_config.state_transform_activation,
        )

        # Decoder
        self.decoder_lstm = LSTMLayer(
            ann.model_config.n2, return_sequences=True
        )
        self.decoder_dense = TimeDistributed(Dense(model_config.dim_out))
        self.compile(
            optimizer=self.optimizer,
            loss=weighted_loss(
                *model_config.loss_weights,
                loss_funcs=model_config.loss_funcs,
            ),
            metrics=model_config.metrics,
        )

    @tf.function
    def decode_sequence(self, encoder_output: tf.Tensor):
        decoder_buffer = tf.zeros(
            (tf.shape(encoder_output)[0], 1, self.model_config.dim_out)
        )

        state_h, state_c = encoder_output, encoder_output
        all_outputs = tf.TensorArray(
            dtype=tf.float32,
            size=self.model_config.seq_len,
            dynamic_size=True,
        )

        for t in tf.range(self.model_config.seq_len):
            decoder_output = self.decoder_lstm(
                decoder_buffer, initial_state=[state_h, state_c]
            )
            decoder_buffer = self.decoder_dense(decoder_output)
            all_outputs = all_outputs.write(t, decoder_buffer)
        return tf.reshape(
            all_outputs.stack(),
            [-1, self.model_config.seq_len, self.model_config.dim_out],
        )

    def call(self, inputs: List[tf.Tensor], training: bool = False):
        if training:
            assert isinstance(inputs, (list, tuple)) and len(inputs) == 2, (
                "LSTM model must be trained with two inputs: "
                "encoder_input and decoder_input.\n"
                f"inputs: {inputs}"
            )
            encoder_input, decoder_input = inputs
            encoder_output = self.encoder(encoder_input)

            # Assuming the encoder output can be used as the initial state for the decoder LSTM.
            state_h = self.state_h_transform(encoder_output)
            state_c = self.state_c_transform(encoder_output)

            decoder_output = self.decoder_lstm(  # type: ignore
                decoder_input, initial_state=[state_h, state_c]
            )
            return self.decoder_dense(decoder_output)

        else:
            encoder_input = inputs
            encoder_output = self.encoder(encoder_input)
            return self.decode_sequence(encoder_output)


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
