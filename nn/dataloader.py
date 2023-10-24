import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Tuple,
    Union,
)
import numpy as np

import pandas as pd
from sklearn.model_selection import KFold

from .config import BaseModelConfig
from .typings import DataLike, ListData, PickleHistory, SingleData
from .utils.logger import ApiLogger

logger = ApiLogger(__name__)


def dump_pickle(
    file_path: os.PathLike, data: Union[PickleHistory, List[PickleHistory]]
) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: os.PathLike) -> PickleHistory:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_pickle_list(
    file_path: os.PathLike,
) -> List[PickleHistory]:
    with open(file_path, "rb") as f:
        return list(pickle.load(f))


def dump_jsonl(file_path: os.PathLike, data: List[Any]) -> None:
    Path(file_path).write_text(
        "\n".join(json.dumps(entry) for entry in data)
    )


def load_jsonl(file_path: os.PathLike) -> List[Dict[str, object]]:
    return [
        json.loads(line) for line in Path(file_path).read_text().splitlines()
    ]


@dataclass
class DataLoader:
    model_config: BaseModelConfig

    train_inputs: DataLike
    train_outputs: SingleData

    train_input_params: Iterable[str]
    train_output_params: Iterable[str]

    def __post_init__(self):
        if isinstance(self.train_inputs, pd.DataFrame) and isinstance(
            self.train_outputs, pd.DataFrame
        ):
            assert (
                self.train_inputs.shape[0] == self.train_outputs.shape[0]
            ), f"{self.train_inputs.shape} != {self.train_outputs.shape}"

            x_columns: List[str] = self.train_inputs.columns.tolist()
            y_columns: List[str] = self.train_outputs.columns.tolist()
            assert isinstance(x_columns, list) and isinstance(
                y_columns, list
            )
            assert set(self.train_input_params) == set(
                x_columns
            ), f"{x_columns} != {self.train_input_params}"
            assert set(self.train_output_params) == set(
                y_columns
            ), f"{y_columns} != {self.train_output_params}"

            logger.debug(
                f"===== Train inputs: {self.train_inputs.shape} ====="
            )
            logger.debug(self.train_inputs.head(48))
            logger.debug(
                f"===== Train outputs: {self.train_outputs.shape} ====="
            )
            logger.debug(self.train_outputs.head(3))

    def dataset_batch_iterator(
        self, batch_size: int = 1
    ) -> Iterator[Tuple[DataLike, DataLike]]:
        xs, ys = self.train_inputs, self.train_outputs
        dataset_sizes = []
        for data in (xs, ys):
            if isinstance(data, (list, tuple)):
                for d in data:
                    dataset_sizes.append(len(d))
            else:
                dataset_sizes.append(len(data))

        dataset_size = int(min(dataset_sizes))  # type: ignore[no-untyped-call]
        for batch_start in range(0, dataset_size, batch_size):
            batch_end = min(dataset_size, batch_start + batch_size)
            if isinstance(xs, (list, tuple)):
                x_batch = [x[batch_start:batch_end] for x in xs]
            else:
                x_batch = xs[batch_start:batch_end]
            yield x_batch, ys[batch_start:batch_end]  # type: ignore[covariant]

    def dataset_kfold_iterator(
        self, n_splits: int = 5
    ) -> Iterator[Tuple[DataLike, SingleData, DataLike, SingleData]]:
        def extract_data_based_on_indices(
            data: ListData, indices: np.ndarray
        ) -> DataLike:
            return [d[indices] for d in data]  # type: ignore[return-value]

        def ensure_list_format(data: DataLike) -> ListData:
            return data if isinstance(data, (list, tuple)) else [data]  # type: ignore[return-value]  # noqa: E501

        xs, ys = ensure_list_format(self.train_inputs), ensure_list_format(
            self.train_outputs
        )

        kf = KFold(n_splits=n_splits)
        for train_indices, test_indices in kf.split(xs[0]):
            x_train, y_train = extract_data_based_on_indices(
                xs, train_indices
            ), extract_data_based_on_indices(ys, train_indices)
            x_test, y_test = extract_data_based_on_indices(
                xs, test_indices
            ), extract_data_based_on_indices(ys, test_indices)

            yield (  # type: ignore[return-value]
                x_train[0] if len(x_train) == 1 else x_train,
                y_train[0] if len(y_train) == 1 else y_train,
                x_test[0] if len(x_test) == 1 else x_test,
                y_test[0] if len(y_test) == 1 else y_test,
            )
