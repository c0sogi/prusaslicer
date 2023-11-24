# flake8: noqa
from dataclasses import asdict
from typing import Optional

import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Layer, MultiHeadAttention
from keras.models import load_model
from keras.optimizers import Adam

from .config import LSTMModelConfig


class EmbeddingAttentionLSTMRegressor(Model):
    def get_config(self):
        config = super().get_config()
        config.update({"model_config": asdict(self.model_config)})
        return config

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            model_config=LSTMModelConfig.from_dict(
                config.pop("model_config")
            ),
            **config,
        )

    def __init__(self, model_config: LSTMModelConfig, **kwargs):
        super().__init__()
        self.optimizer = Adam(learning_rate=model_config.lr)
        if model_config.ann_model_path:
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
            self.feature_extractor = Model(inputs=input_tensor, outputs=x)
            self.feature_extractor.trainable = False
            embedding_dim = ann.model_config.n3
            print("Loaded ANN model:", model_config.ann_model_path)
        else:
            self.feature_extractor = Dense(model_config.dim_in)
            embedding_dim = model_config.dim_in
            print("No ANN model is loaded.")
        self.multihead_attention = MultiHeadAttention(
            num_heads=model_config.num_heads, key_dim=embedding_dim
        )

        # 출력을 원하는 시퀀스 길이로 변환
        self.output_dense = Dense(model_config.seq_len)
        self.reshape_layer = tf.keras.layers.Reshape(
            (model_config.seq_len, 1)
        )
        self.model_config = model_config
        self.compile(
            optimizer=self.optimizer,
            loss=model_config.loss_funcs,
            metrics=model_config.metrics,
        )

    @tf.function
    def call(self, inputs):
        # feature_extractor를 통한 특징 추출
        features = self.feature_extractor(
            inputs
        )  # (batch_size, embedding_dim)

        # 차원 확장
        features = tf.expand_dims(
            features, axis=1
        )  # (batch_size, 1, embedding_dim)

        # MultiHeadAttention 레이어를 이용해 Self-Attention 수행
        attention_output = self.multihead_attention(
            features, features
        )  # (batch_size, 1, embedding_dim)

        # 출력 변환
        output = self.output_dense(attention_output)  # (batch_size, seq_len)
        output = self.reshape_layer(output)
        return output

    # @tf.function
    # def call(self, inputs):
    #     """Attention mechanism"""
    #     batch_size = tf.shape(inputs)[0]
    #     seq_len = self.model_config.seq_len
    #     embedding_dim = self.decoder_lstm.units

    #     # 1. Encoder
    #     q = self.feature_extractor(inputs)  # [batch_size, embedding_size]
    #     v = tf.zeros(
    #         [batch_size, seq_len, embedding_dim]
    #     )  # [batch_size, seq_len, embedding_size]

    #     for step in tf.range(seq_len):
    #         _, attention_weights = self.attention(q, v)  # type: ignore

    #         # Apply the mask
    #         # Create a 1D mask for a single sequence
    #         mask_1d = tf.sequence_mask([step + 1], seq_len, dtype=tf.float32)
    #         mask_1d = tf.squeeze(
    #             mask_1d, axis=0
    #         )  # Squeeze to shape [seq_length]

    #         # Tile the mask to match the batch size and reshape
    #         mask = tf.tile(tf.expand_dims(mask_1d, 0), [batch_size, 1])
    #         mask = tf.expand_dims(
    #             mask, 2
    #         )  # Shape [batch_size, seq_length, 1]
    #         masked_attention_weights = attention_weights * mask

    #         # 각 행의 가중치 합 계산
    #         # 0으로 나누는 것을 방지하기 위해 아주 작은 값을 더함
    #         sum_weights = (
    #             tf.reduce_sum(
    #                 masked_attention_weights, axis=1, keepdims=True
    #             )
    #             + 1e-10
    #         )

    #         # 가중치를 정규화
    #         masked_attention_weights /= sum_weights

    #         # Calculate the context vector
    #         context_vector = masked_attention_weights * v
    #         context_vector = tf.reduce_sum(context_vector, axis=1)
    #         # context_vector와 state_h를 결합

    #         # 인덱스 생성
    #         indices = tf.stack(
    #             [tf.range(batch_size), tf.fill([batch_size], step)], axis=1
    #         )

    #         # v 업데이트
    #         v = tf.tensor_scatter_nd_update(v, indices, context_vector)
    #     return self.output_layer(v)


class BahdanauAttention(Layer):
    def __init__(self, units: int):
        super().__init__()
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

    # def step_fn(states, _):
    #     decoder_state_h, decoder_state_c, decoder_input = states

    #     # # Attention mechanism
    #     # context_vector, _ = self.attention(
    #     #     decoder_state_h, encoder_outputs
    #     # )
    #     # # Reshape context_vector from [batch_size, embedding_size] to [batch_size, 1, embedding_size]
    #     # # Then, combine context vector with current decoder input
    #     # combined_input = tf.concat(
    #     #     [tf.expand_dims(context_vector, 1), decoder_input], axis=-1
    #     # )  # [batch_size, 1, embedding_size * 2]

    #     # Decoder LSTM
    #     (
    #         decoder_output,
    #         decoder_state_h,
    #         decoder_state_c,
    #     ) = self.decoder_lstm(
    #         decoder_input,
    #         initial_state=[decoder_state_h, decoder_state_c],
    #     )
    #     return (decoder_state_h, decoder_state_c, decoder_output)

    # # Using tf.scan for the decoder
    # _, _, outputs = tf.scan(
    #     step_fn,
    #     elems=tf.range(self.model_config.seq_len),
    #     initializer=(initial_state_h, initial_state_c, decoder_input),
    # )  # [seq_len, batch_size, 1, hidden_dim]
    # # 3. Return the sequence of decoder outputs
    # return self.output_layer(
    #     tf.transpose(tf.squeeze(outputs, 2), [1, 0, 2])
    # )
