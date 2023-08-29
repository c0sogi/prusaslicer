from typing import Any, Dict, Optional

import numpy as np
from tensorflow import keras

from .utils.logger import ApiLogger

logger = ApiLogger(__name__)


class AccuracyPerEpoch(keras.callbacks.Callback):
    def __init__(self, print_per_epoch: int, start_epoch: int = 0):
        super().__init__()
        self._ppe = print_per_epoch
        self._epoch = start_epoch

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ):
        self._epoch += 1
        if self._epoch % self._ppe != 0 or logs is None:
            return
        self.print(
            self._epoch,
            rmse=np.sqrt(logs["mse"]) if "mse" in logs else None,
            **logs,
        )

    @staticmethod
    def print(epoch: int, **kwargs: Any) -> None:
        logger.debug(
            f"[Epoch {epoch:^5}]\t"
            + "\t".join(
                f"{key}: {value:.5f}"
                if isinstance(value, (int, float))
                else f"{key}: {value}"
                for key, value in kwargs.items()
            )
        )


EarlyStopping = keras.callbacks.EarlyStopping
