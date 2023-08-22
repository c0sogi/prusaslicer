from tensorflow import keras

from .config import ANNConfig


class ANN(keras.Sequential):
    def __init__(
        self,
        model_config: ANNConfig,
        n1: int,
        n2: int,
        lr: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.add(
            keras.layers.Dense(
                units=model_config.dim_in,
                input_shape=(model_config.train_data.shape[1],),
                activation=activation,
            )
        )
        self.add(keras.layers.Dense(units=n1, activation=activation))
        self.add(keras.layers.Dense(units=n2, activation=activation))
        self.add(keras.layers.Dense(units=model_config.dim_out))
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.mean_absolute_error,
            metrics=model_config.metrics,
        )
