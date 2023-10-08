import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
from sklearn.model_selection import KFold

from .config import BaseModelConfig
from .typings import DataLike, PickleHistory
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

    train_inputs: pd.DataFrame
    train_outputs: pd.DataFrame

    train_input_params: Iterable[str]
    train_output_params: Iterable[str]

    train_inputs_processor: Optional[
        Callable[[pd.DataFrame], DataLike]
    ] = None
    train_outputs_processor: Optional[
        Callable[[pd.DataFrame], DataLike]
    ] = None

    # To be filled

    def __post_init__(self):
        assert (
            self.train_inputs.shape[0] == self.train_outputs.shape[0]
        ), f"{self.train_inputs.shape} != {self.train_outputs.shape}"

        x_columns: List[str] = self.train_inputs.columns.tolist()
        y_columns: List[str] = self.train_outputs.columns.tolist()
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        assert set(self.train_input_params) == set(
            x_columns
        ), f"{x_columns} != {self.train_input_params}"
        assert set(self.train_output_params) == set(
            y_columns
        ), f"{y_columns} != {self.train_output_params}"

        logger.debug(f"===== Train inputs: {self.train_inputs.shape} =====")
        logger.debug(self.train_inputs.head(48))
        logger.debug(
            f"===== Train outputs: {self.train_outputs.shape} ====="
        )
        logger.debug(self.train_outputs.head(3))

    def dataset_batch_iterator(
        self, batch_size: int = 1
    ) -> Iterator[Tuple[DataLike, DataLike]]:
        x_data, y_data = self.train_inputs, self.train_outputs
        dataset_size = min(len(x_data), len(y_data))
        for batch_start in range(0, dataset_size, batch_size):
            batch_end = min(dataset_size, batch_start + batch_size)
            yield self._preprocess_dataset(
                x_data[batch_start:batch_end], y_data[batch_start:batch_end]
            )

    def dataset_kfold_iterator(
        self, n_splits: int = 5
    ) -> Iterator[Tuple[DataLike, DataLike, DataLike, DataLike]]:
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(
            self.train_inputs, self.train_outputs
        ):
            x_train, y_train = self._preprocess_dataset(
                self.train_inputs.iloc[train_index],
                self.train_outputs.iloc[train_index],
            )
            x_test, y_test = self._preprocess_dataset(
                self.train_inputs.iloc[test_index],
                self.train_outputs.iloc[test_index],
            )
            yield x_train, y_train, x_test, y_test

    def get_input_data(self, column_name: str) -> pd.Series:
        return self.train_inputs[column_name]

    def get_output_data(self, column_name: str) -> pd.Series:
        return self.train_outputs[column_name]

    def _preprocess_dataset(
        self, x_data: pd.DataFrame, y_data: pd.DataFrame
    ) -> Tuple[DataLike, DataLike]:
        return (
            self.train_inputs_processor(x_data)
            if self.train_inputs_processor is not None
            else x_data
        ), (
            self.train_outputs_processor(x_data)
            if self.train_outputs_processor is not None
            else y_data
        )
