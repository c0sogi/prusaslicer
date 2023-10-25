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

        freeze_layers = model_config.freeze_layers
        if freeze_layers:
            for layer in freeze_layers:
                getattr(self, layer).trainable = False
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


# class PhysicsInformedANN(ANNFrame):
#     def __init__(self, model_config: ANNModelConfig, **kwargs):
#         # Define model parameters
#         self.model_config = model_config
#         super().__init__(**kwargs)

#         # Define optimizer
#         self.optimizer = Adam(learning_rate=model_config.lr)

#         # Define layers
#         activation = model_config.activation
#         # self.physics_layer = PhysicsInformedLayer(model_config.dim_out)
#         self.dense1 = Dense(units=model_config.n1, activation=activation)
#         self.dense2 = Dense(units=model_config.n2, activation=activation)
#         self.dense3 = Dense(units=model_config.n3, activation=activation)
#         self.dense_out = Dense(units=model_config.dim_out)
#         self.compile(
#             optimizer=self.optimizer,
#             loss=weighted_loss(
#                 *model_config.loss_weights,
#                 loss_funcs=model_config.loss_funcs,
#             ),
#             metrics=model_config.metrics,
#         )

#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         # Combine the physics-informed output and the neural network output
#         return self.dense_out(  # type: ignore
#             self.dense3(
#                 self.dense2(
#                     self.dense1(
#                         inputs,
#                     )
#                 )
#             )
#         )


# # Define the physics-informed layer as a custom Keras layer
# class PhysicsInformedLayer(Layer):
#     def __init__(
#         self,
#         output_dim: int,
#         **kwargs,
#     ):
#         self.output_dim = output_dim

#         super().__init__(**kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(
#             name="kernel",
#             shape=(input_shape[1], self.output_dim),
#             initializer=initializers.RandomNormal(mean=0.0, stddev=1.0),
#             trainable=True,
#         )
#         super().build(input_shape)

#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         return compute_physics(inputs, self.kernel)  # type: ignore


# def compute_physics(inputs: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
#     """Compute the physics-informed mechanical strength and printed quality
#     based on the input parameters and the kernel.

#     The kernel is a 10x2 matrix that contains the weights for each input parameter.
#     The first column corresponds to mechanical strength,
#     and the second column corresponds to printed quality.

#     ### input parameter -> strength / printed quality ###
#     1. bed_temp -> 0.8 / 1.0
#     2. extruder_temp -> 1.0 / 1.0
#     3. layer_thickness -> 1.0 / 1.0
#     4. infill_speed -> 0.8 / 0.8
#     5. density -> 1.0 / 0.5
#     6. thermal_resistance -> 0.3 / 0.0
#     7. impact_strength -> 1.0 / 0.0
#     8. glass_transition_temp -> 0.8 / 0.0
#     9. thermal_conductivity -> 0.2 / 0.0
#     10. linear_thermal_expansion_coefficient -> 0.1 / 0.0
#     """
#     # Cast the input tensor to float32
#     inputs_float = tf.cast(inputs, tf.float32)

#     # For this example, assume that the kernel has shape (10, 2),
#     # and each row corresponds to one of the ten input parameters.

#     # Create a mask for the kernel that "selects" parts of the kernel
#     # based on how much each parameter correlates with mechanical strength
#     # and printed quality. Set value to 1.0 where correlation is strong, and 0.0 where it's weak.
#     mask_mechanical_strength = tf.constant(
#         [0.8, 1.0, 1.0, 0.8, 1.0, 0.3, 1.0, 0.8, 0.2, 0.1], shape=(10, 1)
#     )
#     mask_printed_quality = tf.constant(
#         [1.0, 1.0, 1.0, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], shape=(10, 1)
#     )
#     mask = tf.concat(
#         [mask_mechanical_strength, mask_printed_quality], axis=1
#     )

#     # Element-wise multiply the kernel with the mask
#     masked_kernel = kernel * mask

#     # Use matmul to perform a linear transformation of the inputs
#     outputs = tf.matmul(inputs_float, masked_kernel)
#     return outputs


# class ANN(ModelFrame):
#     def __init__(self, model_config: ModelConfig, **kwargs):
#         # Define model parameters
#         self.model_config = model_config
#         super().__init__(**kwargs)

#         # Define optimizer
#         self.optimizer = Adam(learning_rate=model_config.lr)

#         # Define layers
#         activation = model_config.activation
#         self.dense1 = Dense(units=model_config.n1, activation=activation)
#         self.dense2 = Dense(units=model_config.n2, activation=activation)
#         self.dense3 = Dense(units=model_config.n3, activation=activation)
#         self.dense_out = Dense(units=model_config.dim_out)
#         self.compile(
#             optimizer=self.optimizer,
#             loss=weighted_loss(
#                 *model_config.loss_weights,
#                 loss_funcs=model_config.loss_funcs,
#             ),
#             metrics=model_config.metrics,
#         )

#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         return self.dense_out(  # type: ignore
#             self.dense3(
#                 self.dense2(
#                     self.dense1(
#                         inputs,
#                     )
#                 )
#             )
#         )

# def compute_physics(inputs: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
#     """1. **bed_temp (Bed Temperature)**:
#     - **Influence**: Moderate-High on printed quality, Low-Moderate on mechanical strength.
#     - **Trend**:
#         - **High Value**: A higher bed temperature ensures better adhesion of the first layer to the bed, reducing warping and ensuring a smoother bottom surface.
#         - **Low Value**: Too low of a bed temperature might lead to poor adhesion and increased chances of warping, especially with materials like ABS.
#     - **Reason**: The first layer is critical for the success of the print. A proper bed temperature ensures better adhesion and reduces internal stresses.

#     2. **extruder_temp (Extruder Temperature)**:
#     - **Influence**: High on both printed quality and mechanical strength.
#     - **Trend**:
#         - **High Value**: Overheating can lead to material degradation, stringing, and blobs.
#         - **Low Value**: Too cold and the material might not extrude properly, leading to under-extrusion or weak layer bonding.
#     - **Reason**: The extruder temperature determines how well the filament melts and adheres to the previous layer. Optimal temperature is material-dependent.

#     3. **layer_thickness**:
#     - **Influence**: High on both printed quality and mechanical strength.
#     - **Trend**:
#         - **High Value**: Thicker layers might print faster but can lead to a more visible layer line and possibly reduced detail.
#         - **Low Value**: Thin layers can capture more detail and might provide a smoother finish but take longer to print.
#     - **Reason**: Layer height determines the resolution of the print and the layer-to-layer adhesion.

#     4. **infill_speed**:
#     - **Influence**: Moderate on printed quality, Moderate-High on mechanical strength.
#     - **Trend**:
#         - **High Value**: Faster infill speeds might lead to under-extrusion or weak infill structures.
#         - **Low Value**: Slower infill speeds can ensure better bonding and a stronger internal structure but increase print time.
#     - **Reason**: Speed affects how well the filament is laid down and bonded.

#     5. **density (Infill Density)**:
#     - **Influence**: High on mechanical strength, Low-Moderate on printed quality.
#     - **Trend**:
#         - **High Value**: A denser infill leads to a heavier, stronger print but uses more material.
#         - **Low Value**: Lighter infill might be weaker structurally but is faster to print and uses less material.
#     - **Reason**: Infill provides the internal structure of the print, affecting its overall strength.

#     6. **thermal_resistance**:
#     - **Influence**: Low on printed quality, Moderate on mechanical strength (for specific applications).
#     - **Trend**: This is more about material properties than print settings. A material with high thermal resistance can withstand higher temperatures without deforming.
#     - **Reason**: Some applications require parts that can resist heat.

#     7. **impact_strength**:
#     - **Influence**: Low on printed quality, High on mechanical strength.
#     - **Trend**: Again, this is more about material properties. High impact strength means the material can absorb more energy during sudden forces without breaking.
#     - **Reason**: This is critical for parts that might undergo sudden stresses or impacts.

#     8. **glass_transition_temp**:
#     - **Influence**: Low on printed quality, High on mechanical strength (for specific applications).
#     - **Trend**: The temperature at which the material starts to soften. A higher value means the part can resist higher temperatures before becoming pliable.
#     - **Reason**: Essential for parts used in high-temperature environments.

#     9. **thermal_conductivity**:
#     - **Influence**: Low on printed quality, Low-Moderate on mechanical strength (for specific applications).
#     - **Trend**: It determines how well a material can conduct heat. High conductivity might be desirable for heatsinks, while low conductivity might be preferable for insulating parts.
#     - **Reason**: Specific to the application where heat transfer is a concern.
#     """
#     # Extract features and cast them to float32
#     (
#         bed_temp,
#         extruder_temp,
#         layer_thickness,
#         infill_speed,
#         density,
#         thermal_resistance,
#         impact_strength,
#         glass_transition_temp,
#         thermal_conductivity,
#         linear_thermal_expansion_coefficient,
#     ) = [
#         tf.cast(inputs[:, i : i + 1], tf.float32) for i in INPUT_PARAM_INDICES
#     ]

#     # Mechanical Strength
#     mechanical_strength = (
#         kernel[0, 0] * bed_temp
#         + kernel[1, 0] * extruder_temp
#         + kernel[2, 0] * layer_thickness
#         + kernel[3, 0] * infill_speed
#         + kernel[4, 0] * density
#         + kernel[5, 0] * thermal_resistance
#         + kernel[6, 0] * impact_strength
#         + kernel[7, 0] * glass_transition_temp
#         + kernel[8, 0] * thermal_conductivity
#     )

#     # Printed Quality
#     printed_quality = (
#         kernel[0, 1] * bed_temp
#         + kernel[1, 1] * extruder_temp
#         + kernel[2, 1] * layer_thickness
#         + kernel[3, 1] * infill_speed
#         # Only focusing on the parameters which have a notable influence on print quality.
#         # Others can be added or removed based on more specific domain knowledge.
#     )

#     # Combine the results into a single tensor with shape (None, 2)
#     output = tf.concat([mechanical_strength, printed_quality], axis=1)
#     return output  # type: ignore


# # Utility function to compute the physics-informed mechanical strength
# def compute_physics(inputs: tf.Tensor, kernel) -> tf.Tensor:
#     # Extract features using the provided constants
#     (
#         bed_temp,
#         extruder_temp,
#         layer_thickness,
#         infill_speed,
#         density,
#         thermal_resistance,
#         impact_strength,
#         glass_transition_temp,
#         thermal_conductivity,
#         linear_thermal_expansion_coefficient,
#     ) = [
#         tf.cast(inputs[:, i : i + 1], tf.float32) for i in INPUT_PARAM_INDICES
#     ]
#     print(inputs.shape)

#     # Compute Young's Modulus (E)
#     # Assume Young's Modulus (E) is a function of bed and extruder temperatures
#     # It's a fundamental property in materials science
#     # that quantifies the relationship between stress and strain.
#     # It's often affected by temperature,
#     # which is why I included bed and extruder temperatures as factors.
#     E = bed_temp * kernel[0] + extruder_temp * kernel[1]

#     # Compute Strain (epsilon)
#     # Strain (epsilon) could be simplified as layer_thickness / infill_speed
#     # In a simplified model, the strain can be thought of as
#     # proportional to the layer thickness and inversely proportional to the infill speed.
#     # This is a simplified assumption;
#     # in reality, the strain would depend on various other factors
#     # including material properties and the object's geometry.
#     epsilon = layer_thickness / infill_speed  # type: ignore

#     # Compute and return mechanical strength (Simplified physics equation for Mechanical Strength)
#     # Factors like density, thermal resistance, and impact strength
#     # are fundamental material properties that often influence mechanical properties.
#     # In the example, they are combined in a simplified manner to affect the mechanical strength.
#     return (
#         E
#         * epsilon
#         * (
#             density  # type: ignore
#             * thermal_resistance
#             * impact_strength
#             * glass_transition_temp
#             * thermal_conductivity
#             * linear_thermal_expansion_coefficient
#         )
#     )
