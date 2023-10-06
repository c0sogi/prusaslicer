# flake8: noqa
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, TensorDataset

from .config import N_EXPERIMENTS as K
from .config import N_HIDDEN_LAYERS as H
from .config import N_INPUT_PARAMS as N
from .config import N_OUTPUT_PARAMS as M
from .network import NeuralNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class CheckPoint(TypedDict):
    epoch: int
    model_state_dict: dict
    optimizer_state_dict: dict
    best_loss_val: float


@dataclass
class Trainer:
    model: NeuralNet
    optimizer: optim.Optimizer
    criterion: _Loss
    batch_size: int
    model_path: os.PathLike = Path("./checkpoint/model.pt")
    checkpoint_prefix: str = "best"
    _best_loss_val: float = field(init=False, default=np.inf)
    _epochs: int = field(init=False, default=0)
    _model_path_parent: Path = Path(model_path).parent.resolve()
    _model_path_parent.mkdir(parents=True, exist_ok=True)
    _model_path_stem: str = Path(model_path).stem
    _model_path_suffix: str = Path(model_path).suffix

    def __post_init__(self) -> None:
        """Load the model checkpoint if it exists."""
        try:
            best_checkpoint_path = max(
                self._model_path_parent.glob(
                    f"{self.checkpoint_prefix}_{self._model_path_stem}_*{self._model_path_suffix}"
                )
            )
        except ValueError:
            logger.warning(
                f"No checkpoint found at {self.checkpoint_path}, starting from scratch"
            )
        else:
            logger.info(f"Loading checkpoint from {best_checkpoint_path}")
            self.load_checkpoint(best_checkpoint_path)

    @property
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint file."""
        return (
            self._model_path_parent
            / f"{self.checkpoint_prefix}_{self._model_path_stem}_{self._epochs}{self._model_path_suffix}"
        )

    def train(
        self,
        inputs: npt.NDArray[np.float32],
        outputs: npt.NDArray[np.float32],
        epochs: int,
        checkpoint_interval: Optional[int] = 100,
    ) -> None:
        """Train the model."""
        dataloader = self._dataset_batch(inputs, outputs, batch_size=K)

        # Training Loop
        loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        start_epoch = self._epochs
        max_epoch = self._epochs + epochs
        for epoch in range(start_epoch, max_epoch):
            for batch_inputs, batch_outputs in dataloader:
                # Forward pass
                loss = self.criterion.forward(
                    model.forward(batch_inputs.to(DEVICE)),
                    batch_outputs.to(DEVICE),
                )  # type: torch.Tensor

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                loss_val = loss.item()  # type: float
                self._epochs = epoch + 1
                if loss_val < self._best_loss_val:
                    self._best_loss_val = loss_val
                    self.save_checkpoint()
                logger.info(
                    f"Epoch [{self._epochs}/{max_epoch}], Loss: {loss_val:.4f}"
                )

    def save_checkpoint(self) -> None:
        """Save the model checkpoint."""
        torch.save(
            CheckPoint(
                epoch=self._epochs,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                best_loss_val=self._best_loss_val,
            ),
            self.checkpoint_path,
        )
        logger.info(f"Saved checkpoint at {self.checkpoint_path}")

    def load_checkpoint(self, path: os.PathLike) -> None:
        """Load the model checkpoint."""
        checkpoint = torch.load(path)  # type: CheckPoint
        self._epochs = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._best_loss_val = checkpoint["best_loss_val"]
        logger.info(
            f"Loaded checkpoint from {path} with best loss value {self._best_loss_val:.4f}"
        )

    @staticmethod
    def _dataset_batch(
        inputs: npt.NDArray[np.float32],
        outputs: npt.NDArray[np.float32],
        batch_size: int,
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield batches of data."""
        assert inputs.shape[0] == outputs.shape[0] == K, "Invalid data size"
        assert inputs.shape[0] % batch_size == 0, "Invalid batch size"
        return DataLoader(
            TensorDataset(
                torch.tensor(inputs, dtype=torch.float32, device=DEVICE),
                torch.tensor(outputs, dtype=torch.float32, device=DEVICE),
            ),
            batch_size=batch_size,
            shuffle=True,
        )


if __name__ == "__main__":
    # Replace these arrays with your actual data
    input_data = np.random.rand(K, N).astype(
        np.float32
    )  # k samples of n-dimensional input
    output_data = np.random.rand(K, M).astype(
        np.float32
    )  # k samples of m-dimensional output

    model = NeuralNet(N, H, M).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trainer = Trainer(model, optimizer, criterion, K)
    trainer.train(input_data, output_data, epochs=10000)
