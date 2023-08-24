from typing import Any, Optional

import numpy as np
from tensorflow import keras

from .logger import ApiLogger

logger = ApiLogger(__name__)


class AccuracyPerEpoch(keras.callbacks.Callback):
    def __init__(self, print_per_epoch: int):
        super().__init__()
        self._ppe = print_per_epoch

    def on_epoch_end(
        self, epoch: int, logs: Optional[dict[str, float]] = None
    ):
        if epoch % self._ppe != 0 or logs is None:
            return
        self.print(
            epoch,
            rsme=np.sqrt(logs["mse"]) if "mse" in logs else None,
            mae=logs["mae"] if "mae" in logs else None,
            mape=logs["mape"] if "mape" in logs else None,
            val_rsme=np.sqrt(logs["val_mse"]) if "val_mse" in logs else None,
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
