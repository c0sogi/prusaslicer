# flake8: noqa
from dataclasses import asdict
from typing import Dict, Optional, Union

import tensorflow as tf
from keras import Model, Sequential, initializers
from keras.layers import Dense, Lambda, Layer
from keras.losses import Loss, mean_absolute_error
from keras.optimizers import Adam
from keras.src.engine import data_adapter

from .config import ModelConfig
from .schemas import INPUT_PARAM_INDICES


# Define the physics-informed layer as a custom Keras layer
class PhysicsInformedLayer(Layer):
    def __init__(self, output_dim: int, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], self.output_dim),
            initializer=initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return compute_mechanical_strength(inputs, self.kernel)

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


class PhysicsInformedANN(Model):
    def __init__(self, model_config: ModelConfig, **kwargs):
        # Define model parameters
        self.model_config = model_config
        super().__init__(**kwargs)

        # Define optimizer
        self.optimizer = Adam(learning_rate=model_config.lr)

        # Define layers
        activation = model_config.activation
        self.physics_layer = PhysicsInformedLayer(model_config.dim_out)
        self.dense1 = Dense(units=model_config.n1, activation=activation)
        self.dense2 = Dense(units=model_config.n2, activation=activation)
        self.dense3 = Dense(units=model_config.n3, activation=activation)
        self.dense_out = Dense(units=model_config.dim_out)
        self.compile(
            optimizer=self.optimizer,
            loss=mean_absolute_error,
            metrics=model_config.metrics,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Combine the physics-informed output and the neural network output
        return self.physics_layer(inputs) + self.dense_out(  # type: ignore
            self.dense3(
                self.dense2(
                    self.dense1(
                        inputs,
                    )
                )
            )
        )

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
            model_config=ModelConfig.from_dict(config.pop("model_config")),
            **config,
        )


# Utility function to compute the physics-informed mechanical strength
def compute_mechanical_strength(inputs: tf.Tensor, kernel) -> tf.Tensor:
    # Extract features using the provided constants
    (
        bed_temp,
        extruder_temp,
        layer_thickness,
        infill_speed,
        density,
        thermal_resistance,
        impact_strength,
        glass_transition_temp,
        thermal_conductivity,
        linear_thermal_expansion_coefficient,
    ) = [
        tf.cast(inputs[:, i : i + 1], tf.float32) for i in INPUT_PARAM_INDICES
    ]

    # Compute Young's Modulus (E)
    # Assume Young's Modulus (E) is a function of bed and extruder temperatures
    # It's a fundamental property in materials science
    # that quantifies the relationship between stress and strain.
    # It's often affected by temperature,
    # which is why I included bed and extruder temperatures as factors.
    E = bed_temp * kernel[0] + extruder_temp * kernel[1]

    # Compute Strain (epsilon)
    # Strain (epsilon) could be simplified as layer_thickness / infill_speed
    # In a simplified model, the strain can be thought of as
    # proportional to the layer thickness and inversely proportional to the infill speed.
    # This is a simplified assumption;
    # in reality, the strain would depend on various other factors
    # including material properties and the object's geometry.
    epsilon = layer_thickness / infill_speed  # type: ignore

    # Compute and return mechanical strength (Simplified physics equation for Mechanical Strength)
    # Factors like density, thermal resistance, and impact strength
    # are fundamental material properties that often influence mechanical properties.
    # In the example, they are combined in a simplified manner to affect the mechanical strength.
    return (
        E
        * epsilon
        * (
            density  # type: ignore
            * thermal_resistance
            * impact_strength
            * glass_transition_temp
            * thermal_conductivity
            * linear_thermal_expansion_coefficient
        )
    )


# Original: keras.losses.mean_absolute_error
def physics_informed_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, physics_output: tf.Tensor
) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)  # type: ignore
    y_pred_0 = tf.cast(y_pred[0], tf.float32)
    y_pred_1 = tf.cast(y_pred[1], tf.float32)
    physics_output = tf.cast(physics_output, tf.float32)  # type: ignore
    # Loss sum: predicted loss + physics loss
    return tf.reduce_mean(tf.square(y_true - y_pred_0)) + tf.reduce_mean(
        tf.square(y_pred_1 - physics_output)
    )


class ANN(Sequential):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        n3: Optional[int] = None,
        lr: Optional[float] = None,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if (
            model_config is not None
            and n1 is not None
            and n2 is not None
            and n3 is not None
            and lr is not None
        ):
            self.optimizer = Adam(learning_rate=lr)
            self.add(Dense(units=n1, activation=activation))
            self.add(Dense(units=n2, activation=activation))
            self.add(Dense(units=n3, activation=activation))
            self.add(Dense(units=model_config.dim_out))

            self.compile(
                optimizer=self.optimizer,
                loss=mean_absolute_error,
                metrics=model_config.metrics,
            )
