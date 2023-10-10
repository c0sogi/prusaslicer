# flake8: noqa
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Union

import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, TimeDistributed, Layer
from keras.models import load_model
from keras.optimizers import Adam
from keras.src.engine import data_adapter

from .config import LSTMModelConfig
from .losses import weighted_loss


class EmbeddingAttentionLSTMRegressor(tf.keras.Model):
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
        super().__init__()
        self.optimizer = Adam(learning_rate=model_config.lr)
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
        self.embedding = Model(inputs=input_tensor, outputs=x)
        self.embedding.trainable = False
        self.rnn = LSTM(
            ann.model_config.n2, return_sequences=True, return_state=True
        )
        self.attention = BahdanauAttention(self.rnn.units)
        self.output_layer = Dense(1)
        self.model_config = model_config
        self.compile(
            optimizer=self.optimizer,
            loss=model_config.loss_funcs,
            metrics=model_config.metrics,
        )

    @tf.function
    def call(self, inputs):
        x = self.embedding(inputs)
        batch_size = tf.shape(inputs)[0]
        state_h = tf.zeros((batch_size, self.rnn.units))
        state_c = tf.zeros((batch_size, self.rnn.units))
        initial_states_outputs = (
            (state_h, state_c),
            tf.zeros((batch_size, 1)),
        )

        def step_fn(states_outputs, _):
            (state_h, state_c), _ = states_outputs
            context_vector, _ = self.attention(state_h, x)
            _, state_h, state_c = self.rnn(
                tf.expand_dims(context_vector, 1),
                initial_state=[state_h, state_c],
            )
            output = self.output_layer(state_h)
            return ((state_h, state_c), output)

        _, outputs = tf.scan(
            step_fn,
            elems=tf.range(self.model_config.seq_len),
            initializer=initial_states_outputs,
        )

        return tf.squeeze(outputs, axis=2)


class BahdanauAttention(Layer):
    def __init__(self, units: int):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    @tf.function
    def call(self, query: tf.Tensor, values: tf.Tensor):
        # Expand dimension of query to (batch_size, 1, hidden_size)
        query_with_time_axis = tf.expand_dims(query, 1)

        # Calculate the attention scores
        score = self.V(
            tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values))  # type: ignore
        )

        # Calculate attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Calculate the context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


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
