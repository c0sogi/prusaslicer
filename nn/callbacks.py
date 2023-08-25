from typing import Any, Dict, Optional

import numpy as np
from tensorflow import keras

from .logger import ApiLogger

logger = ApiLogger(__name__)


class AccuracyPerEpoch(keras.callbacks.Callback):
    def __init__(self, print_per_epoch: int):
        super().__init__()
        self._ppe = print_per_epoch
        self._epoch = 0

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ):
        self._epoch += 1
        if self._epoch % self._ppe != 0 or logs is None:
            return
        self.print(
            self._epoch,
            rmse=np.sqrt(logs["mse"]) if "mse" in logs else None,
            mae=logs["mae"] if "mae" in logs else None,
            mape=logs["mape"] if "mape" in logs else None,
            val_rmse=np.sqrt(logs["val_mse"]) if "val_mse" in logs else None,
            val_mae=logs["val_mae"] if "val_mae" in logs else None,
            val_mape=logs["val_mape"] if "val_mape" in logs else None,
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


class EarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience: int):
        super().__init__()
        self._patience = patience
        self._best = np.Inf
        self._count = 0

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ):
        if logs is None:
            return
        current = logs["val_mse"]
        if current < self._best:
            self._best = current
            self._count = 0
        else:
            self._count += 1
            if self._count >= self._patience:
                self.model.stop_training = True  # type: ignore
