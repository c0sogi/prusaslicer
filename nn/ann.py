from typing import Optional
import numpy as np
from tensorflow import keras

from .config import ModelConfig


class ANN(keras.Sequential):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        n0: Optional[int] = None,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        lr: Optional[float] = None,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if (
            model_config is not None
            and n1 is not None
            and n2 is not None
            and lr is not None
        ):
            self.optimizer = keras.optimizers.Adam(learning_rate=lr)
            self.add(
                keras.layers.Dense(
                    units=n0,
                    input_shape=(model_config.dim_in,),
                    activation=activation,
                )
            )
            self.add(keras.layers.Dense(units=n1, activation=activation))
            self.add(keras.layers.Dense(units=n2, activation=activation))
            self.add(keras.layers.Dense(units=model_config.dim_out))
            self.compile(
                optimizer=self.optimizer,
                loss=keras.losses.mean_absolute_error,
                metrics=model_config.metrics,
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)
