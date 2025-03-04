# # flake8: noqa
# from dataclasses import asdict
# from typing import Dict, Optional, Union

# import tensorflow as tf
# from keras import Model
# from keras.layers import (
#     LSTM,
#     Dense,
#     Dropout,
#     LayerNormalization,
#     TimeDistributed,
# )
# from keras.optimizers import Adam
# from keras.regularizers import l1, l1_l2, l2
# from keras.src.engine import data_adapter

# from .config import ANNModelConfig, LSTMModelConfig
# from .losses import weighted_loss


# class StructureFrame(Model):
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


# class StructuredLSTM(StructureFrame):
#     @classmethod
#     def from_config(cls, config):
#         return cls(
#             model_config=LSTMModelConfig.from_dict(
#                 config.pop("model_config")
#             ),
#             **config,
#         )

#     def __init__(self, model_config: LSTMModelConfig, **kwargs):
#         # Define model parameters
#         self.model_config = model_config
#         super().__init__(**kwargs)

#         # Define optimizer
#         self.optimizer = Adam(learning_rate=model_config.lr)

#         # Encoder
#         self.encoder_dense = tf.keras.layers.Dense(
#             model_config.lstm_units, activation=model_config.activation
#         )
#         self.encoder_lstm = LSTM(model_config.lstm_units, return_state=True)

#         # Decoder
#         self.decoder_lstm = LSTM(
#             model_config.lstm_units, return_sequences=True
#         )
#         self.decoder_dense = TimeDistributed(Dense(model_config.dim_out))
#         self.compile(
#             optimizer=self.optimizer,
#             loss=weighted_loss(
#                 *model_config.loss_weights,
#                 loss_funcs=model_config.loss_funcs,
#             ),
#             metrics=model_config.metrics,
#         )

#     def call(
#         self,
#         data: tf.Tensor,
#         encoder_input: Optional[tf.Tensor] = None,
#         training: bool = False,
#     ):
#         if encoder_input is not None:
#             encoder_output = encoder_input
#         else:
#             encoder_output, state_h, state_c = self.encoder_lstm(
#                 self.encoder_dense(data)
#             )  # type: ignore

#         # ... rest of the method remains the same ...

#     def call(self, inputs: tf.Tensor, training: bool = False):
#         if training:
#             encoder_input, decoder_input = inputs
#             encoder_output, state_h, state_c = self.encoder_lstm(
#                 self.encoder_dense(encoder_input)
#             )  # type: ignore
#             decoder_output, _, _ = self.decoder_lstm(
#                 decoder_input, initial_state=[state_h, state_c]
#             )  # type: ignore
#             return self.decoder_dense(decoder_output)
#         else:
#             encoder_input = inputs
#             encoder_output, state_h, state_c = self.encoder_lstm(
#                 self.encoder_dense(encoder_input)
#             )  # type: ignore
#             decoder_seq = tf.zeros(
#                 (tf.shape(encoder_input)[0], 1, self.model_config.dim_out)
#             )  # Assuming the output dimension is model_config.dim_out

#             all_outputs = []

#             for _ in range(self.model_config.seq_len):
#                 decoder_output, state_h, state_c = self.decoder_lstm(
#                     decoder_seq, initial_state=[state_h, state_c]
#                 )  # type: ignore
#                 decoder_seq = self.decoder_dense(decoder_output)
#                 all_outputs.append(decoder_seq)

#             return tf.concat(all_outputs, axis=1)


# class StructuredANN(StructureFrame):
#     @classmethod
#     def from_config(cls, config):
#         return cls(
#             model_config=ANNModelConfig.from_dict(
#                 config.pop("model_config")
#             ),
#             **config,
#         )

#     def __init__(self, model_config: ANNModelConfig, **kwargs):
#         # Define model parameters
#         self.model_config = model_config
#         super().__init__(**kwargs)

#         # Define optimizer
#         self.optimizer = Adam(learning_rate=model_config.lr)

#         # Define regularization
#         if model_config.l1_reg and model_config.l2_reg:
#             kernel_regularizer = l1_l2(
#                 l1=model_config.l1_reg, l2=model_config.l2_reg
#             )
#         elif model_config.l1_reg:
#             kernel_regularizer = l1(model_config.l1_reg)
#         elif model_config.l2_reg:
#             kernel_regularizer = l2(model_config.l2_reg)
#         else:
#             kernel_regularizer = None

#         # Define layers
#         activation = model_config.activation
#         self.dense1 = Dense(
#             units=model_config.n1,
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

#         self.dense2 = Dense(
#             units=model_config.n2,
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
#         self.dense3 = Dense(
#             units=model_config.n3,
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
#         self.dense_out = Dense(units=model_config.dim_out)
#         self.compile(
#             optimizer=self.optimizer,
#             loss=weighted_loss(
#                 *model_config.loss_weights,
#                 loss_funcs=model_config.loss_funcs,
#             ),
#             metrics=model_config.metrics,
#         )

#     def call(self, inputs: tf.Tensor, training: bool = False):
#         x = self.dense1(inputs)
#         if self.norm1 is not None:
#             x = self.norm1(x)
#         if self.dropout1 is not None:
#             x = self.dropout1(x, training=training)
#         x = self.dense2(x)
#         if self.norm2 is not None:
#             x = self.norm2(x)
#         if self.dropout2 is not None:
#             x = self.dropout2(x, training=training)
#         x = self.dense3(x)
#         if self.norm3 is not None:
#             x = self.norm3(x)
#         if self.dropout3 is not None:
#             x = self.dropout3(x, training=training)
#         return self.dense_out(x)


# class CombinedModel(Model):
#     def __init__(
#         self, ann_config: ANNModelConfig, lstm_config: LSTMModelConfig
#     ):
#         super().__init__()
#         self.structured_ann = StructuredANN(ann_config)
#         self.structured_lstm = StructuredLSTM(lstm_config)

#     def call(self, inputs: tf.Tensor, training: bool = False):
#         # Pass inputs through StructuredANN up to the second dense layer
#         ann_output = self.structured_ann.dense2(
#             self.structured_ann.dense1(inputs)
#         )

#         # Use the ANN's second dense layer's output as encoder input for StructuredLSTM
#         lstm_output = self.structured_lstm(
#             inputs, encoder_input=ann_output, training=training
#         )

#         return lstm_output
