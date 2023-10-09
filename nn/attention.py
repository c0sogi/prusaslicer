# # flake8: noqa
# from dataclasses import asdict
# from typing import Dict, List, Union

# import tensorflow as tf
# from keras import Model
# from keras.layers import Dense, Flatten, Input, MultiHeadAttention
# from keras.models import load_model
# from keras.optimizers import Adam
# from keras.src.engine import data_adapter

# from .config import AttentionModelConfig
# from .losses import weighted_loss


# class AttentionFrame(Model):
#     def train_step(
#         self, data: tf.Tensor
#     ) -> Dict[str, Union[float, tf.Tensor]]:
#         (
#             x,
#             y,
#             sample_weight,
#         ) = data_adapter.unpack_x_y_sample_weight(data)

#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
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

#     def get_config(self):
#         config = super().get_config()
#         config.update({"model_config": asdict(self.model_config)})
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(
#             model_config=AttentionModelConfig.from_dict(
#                 config.pop("model_config")
#             ),
#             **config,
#         )


# class Attention(AttentionFrame):
#     def __init__(self, model_config: AttentionModelConfig, **kwargs):
#         # Define model parameters
#         self.model_config = model_config
#         super().__init__(**kwargs)

#         # Define optimizer
#         self.optimizer = Adam(learning_rate=model_config.lr)

#         # Input Layer
#         self.input_layer = Input(shape=(model_config.dim_in,))

#         # Expand dimensions for attention mechanism
#         # (to treat non-series data as a sequence)
#         # - shape: (batch_size, 1, input_dim)
#         self.expanded_input = tf.expand_dims(self.input_layer, axis=1)

#         # Multi-Head Attention
#         # - shape: (batch_size, 1, input_dim)
#         self.attention_out = MultiHeadAttention(
#             num_heads=8, key_dim=model_config.dim_in
#         )(
#             query=self.expanded_input,
#             key=self.expanded_input,
#             value=self.expanded_input,
#         )

#         # Flatten the output for the dense layer
#         # - shape: (batch_size, input_dim)
#         self.flattened_attention_out = Flatten()(self.attention_out)

#         # Encoder
#         ann = load_model(model_config.ann_model_path)
#         assert isinstance(ann, Model), type(ann)
#         encoder_input = ann.layers[0]
#         encoder_out = ann.layers[-2]
#         self.encoder = Model(
#             inputs=encoder_input.input, outputs=encoder_out.output
#         )
#         self.encoder.trainable = False

#         # Decoder
#         self.decoder_dense = Dense(model_config.seq_len)(
#             self.flattened_attention_out
#         )
#         self.compile(
#             optimizer=self.optimizer,
#             loss=weighted_loss(
#                 *model_config.loss_weights,
#                 loss_funcs=model_config.loss_funcs,
#             ),
#             metrics=model_config.metrics,
#         )

#     def call(self, inputs: List[tf.Tensor], training: bool = False):
#         if training:
#             assert isinstance(inputs, (list, tuple)) and len(inputs) == 2, (
#                 "LSTM model must be trained with two inputs: "
#                 "encoder_input and decoder_input.\n"
#                 f"inputs: {inputs}"
#             )
#             encoder_input, decoder_input = inputs
#             encoder_output = self.encoder(encoder_input)
#             _, state_h, state_c = self.encoder_lstm(encoder_output)
#             decoder_output = self.decoder_lstm(
#                 decoder_input, initial_state=[state_h, state_c]
#             )
#             return self.decoder_dense(decoder_output)
#         else:
#             encoder_input = inputs
#             decoder_seq = tf.zeros(
#                 (tf.shape(encoder_input)[0], 1, self.model_config.dim_out)
#             )  # Assuming the output dimension is model_config.dim_out

#             all_outputs = tf.TensorArray(
#                 dtype=tf.float32,
#                 size=self.model_config.seq_len,
#                 dynamic_size=False,
#                 infer_shape=True,
#             )

#             for t in tf.range(self.model_config.seq_len):
#                 decoder_output = self.decoder_lstm(
#                     decoder_seq, initial_state=[state_h, state_c]
#                 )
#                 decoder_seq = self.decoder_dense(decoder_output)
#                 all_outputs = all_outputs.write(t, decoder_seq)

#             return tf.concat(all_outputs.stack(), axis=1)


# def test_multi_head_attention():
#     # Input Layer
#     input_layer = Input(shape=(input_dim,))

#     # Expand dimensions for attention mechanism
#     # (to treat non-series data as a sequence)
#     expanded_input = tf.expand_dims(input_layer, axis=1)

#     # Multi-Head Attention
#     attention_out = MultiHeadAttention(num_heads=8, key_dim=input_dim)(
#         query=expanded_input, key=expanded_input, value=expanded_input
#     )

#     # Flatten the output for the dense layer
#     flattened_attention_out = Flatten()(attention_out)

#     # Some dense layers
#     dense1 = Dense(64, activation="relu")(flattened_attention_out)
#     dense2 = Dense(32, activation="relu")(dense1)

#     # Output layer
#     output_layer = Dense(seq_len)(dense2)

#     model = Model(inputs=input_layer, outputs=output_layer)

#     # Compile the model
#     model.compile(optimizer="adam", loss="mse")
#     return model


# # # Sample Data
# # model = test_multi_head_attention()
# # num_samples = 1000
# # X_train = np.random.rand(
# #     num_samples, input_dim
# # )  # Sample non-series input data
# # Y_train = np.random.rand(
# #     num_samples, seq_len
# # )  # Sample time series output data

# # # Train the model
# # model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

# # # Inferencing
# # sample_input = np.array([[0.5] * input_dim])  # A sample input
# # predicted_time_series = model.predict(sample_input)

# # print(predicted_time_series)
