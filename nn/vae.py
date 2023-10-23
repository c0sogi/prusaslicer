from typing import List, Optional
import numpy as np
import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        dense_units: tuple,
        input_shape: tuple,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.dense_units = sorted(dense_units)
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=input_shape)]
            + [
                tf.keras.layers.Dense(units=unit, activation=tf.nn.relu)
                for unit in reversed(self.dense_units)
            ]
            + [tf.keras.layers.Dense(units=latent_dim + latent_dim)]
        )

        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(latent_dim,))]
            + [
                tf.keras.layers.Dense(units=unit, activation=tf.nn.relu)
                for unit in self.dense_units
            ]
            + [tf.keras.layers.Dense(units=input_shape[-1])]
        )
        if len(input_shape) == 1:
            self.reduce_axes = 1
        else:
            self.reduce_axes = list(range(1, len(input_shape) + 1))

    @tf.function
    def sample(self, eps: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Sample from latent distribution.
        Args:
            eps: Tensor of shape (batch_size, latent_dim)
        Returns:
            Tensor of shape (batch_size, *input_shape)"""

        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        else:
            eps = eps

        return self.decode(eps, apply_sigmoid=True)  # type: ignore

    @tf.function
    def encode(self, x: tf.Tensor) -> List[tf.Tensor]:
        """Return mean and log variance of latent distribution.
        Args:
            x: Tensor of shape (batch_size, *input_shape)
        Returns:
            mean: Tensor of shape (batch_size, latent_dim)
            logvar: Tensor of shape (batch_size, latent_dim)"""
        return tf.split(
            self.encoder(x),
            num_or_size_splits=2,
            axis=1,
        )  # type: ignore

    @tf.function
    def reparameterize(
        self, mean: tf.Tensor, logvar: tf.Tensor
    ) -> tf.Tensor:
        """Reparameterization trick to sample from N(mean, var) from
        N(0,1).
        Args:
            mean: Tensor of shape (batch_size, latent_dim)
            logvar: Tensor of shape (batch_size, latent_dim)
        Returns:
            Tensor of shape (batch_size, latent_dim)"""
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    @tf.function
    def decode(self, z: tf.Tensor, apply_sigmoid: bool = False):
        """Decode latent vector.
        Args:
            z: Tensor of shape (batch_size, latent_dim)
            apply_sigmoid: Whether to apply sigmoid activation function
        Returns:
                Tensor of shape (batch_size, *input_shape)"""
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def log_normal_pdf(
        self,
        sample: tf.Tensor,
        mean: tf.Tensor,
        logvar: tf.Tensor,
        raxis: int = 1,
    ) -> tf.Tensor:
        """Compute log pdf of normal distribution.
        Args:
            sample: Tensor of shape (batch_size, latent_dim)
            mean: Tensor of shape (batch_size, latent_dim)
            logvar: Tensor of shape (batch_size, latent_dim)
            raxis: Axis to sum over
        Returns:
            Tensor of shape (batch_size,)"""
        log2pi = tf.math.log(2.0 * tf.constant(np.pi))
        return tf.reduce_sum(
            -0.5
            * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    @tf.function
    def compute_loss(self, x: tf.Tensor) -> tf.Tensor:
        """Compute loss.
        Args:
            x: Tensor of shape (batch_size, *input_shape)
        Returns:
            Tensor of shape (batch_size,)"""
        mean, logvar = self.encode(x)  # type: ignore
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x
        )
        logpx_z = -tf.reduce_sum(cross_ent, axis=1)
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(self, x: tf.Tensor):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        # # Gradient clipping by norm
        # clipped_gradients = [
        #     tf.clip_by_norm(grad, 0.0000001) for grad in gradients
        # ]

        # optimizer.apply_gradients(
        #     zip(clipped_gradients, self.trainable_variables)
        # )
