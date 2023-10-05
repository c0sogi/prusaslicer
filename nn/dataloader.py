import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Tuple,
    Type,
    Union,
    get_args,
)

import pandas as pd
from sklearn.model_selection import KFold

from nn.utils.logger import ApiLogger
from nn.utils.raw_data import read_overall_table

from .config import BaseModelConfig
from .schemas import (
    ANNInputParams,
    ANNOutputParams,
    CNNInputParams,
    CNNOutputParams,
    PickleHistory,
)

PathLike = Union[str, Path]
logger = ApiLogger(__name__)


def dump_pickle(
    file_path: PathLike, data: Union[PickleHistory, List[PickleHistory]]
) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: PathLike) -> PickleHistory:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_pickle_list(
    file_path: PathLike,
) -> List[PickleHistory]:
    with open(file_path, "rb") as f:
        return list(pickle.load(f))


def dump_jsonl(file_path: PathLike, data: List[Any]) -> None:
    Path(file_path).write_text(
        "\n".join(json.dumps(entry) for entry in data)
    )


def load_jsonl(file_path: PathLike) -> List[Dict[str, object]]:
    return [
        json.loads(line) for line in Path(file_path).read_text().splitlines()
    ]


@dataclass
class BaseDataLoader:
    model_config: BaseModelConfig

    # To be filled
    train_data: pd.DataFrame = field(init=False, repr=False)
    train_label: pd.DataFrame = field(init=False, repr=False)
    input_params_type: Type = field(init=False, repr=False)
    output_params_type: Type = field(init=False, repr=False)
    raw_data_reader: Callable[
        [os.PathLike], Tuple[pd.DataFrame, pd.DataFrame]
    ] = field(init=False, repr=False)

    def __post_init__(self):
        # Input 및 Output DataFrame으로 분리
        self.input_data, self.output_data = self.raw_data_reader(
            Path(self.model_config.input_path)
        )
        assert (
            self.input_data.shape[0] == self.output_data.shape[0]
        ), "데이터 개수 불일치"

        logger.debug(
            f"===== Input Data: {self.input_data.shape} =====\n{self.input_data.head(3)}"  # noqa: E501
        )
        logger.debug(
            f"===== Output Data: {self.output_data.shape} =====\n{self.output_data.head(3)}"  # noqa: E501
        )
        x_params = list(get_args(self.input_params_type))
        x_columns: List[str] = self.input_data.columns.tolist()
        y_params = list(get_args(self.output_params_type))
        y_columns: List[str] = self.output_data.columns.tolist()
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        assert set(x_params) == set(x_columns), f"{x_columns} != {x_params}"
        assert set(y_params) == set(y_columns), f"{y_columns} != {y_params}"
        self.output_column_names = y_columns
        self.train_data = pd.concat(
            [self.get_input_data(column_name) for column_name in x_columns],
            axis=1,
        ).astype(float)
        self.train_label = pd.concat(
            [self.get_input_data(column_name) for column_name in y_columns],
            axis=1,
        ).astype(float)
        logger.debug(f"===== Train Data: {self.train_data.shape} =====")
        logger.debug(self.train_data.head(48))
        logger.debug(f"===== Train Label: {self.train_label.shape} =====")
        logger.debug(self.train_label.head(3))

    def dataset_batch_iterator(
        self, batch_size: int = 1
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        x_data, y_data = self.train_data, self.train_label
        dataset_size = min(len(x_data), len(y_data))
        for batch_start in range(0, dataset_size, batch_size):
            batch_end = min(dataset_size, batch_start + batch_size)
            yield x_data[batch_start:batch_end], y_data[
                batch_start:batch_end
            ]

    def dataset_kfold_iterator(
        self, n_splits: int = 5
    ) -> Iterator[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ]:
        kf = KFold(n_splits=n_splits)
        x_data, y_data = self.train_data, self.train_label
        for train_index, test_index in kf.split(x_data, y_data):
            x_train, x_test = (
                x_data.iloc[train_index],
                x_data.iloc[test_index],
            )
            y_train, y_test = (
                y_data.iloc[train_index],
                y_data.iloc[test_index],
            )
            yield x_train, y_train, x_test, y_test

    def get_input_data(self, column_name: str) -> pd.Series:
        return self.input_data[column_name]

    def get_output_data(self, column_name: str) -> pd.Series:
        return self.output_data[column_name]


@dataclass
class DataLoaderANN(BaseDataLoader):
    raw_data_reader: Callable[
        [os.PathLike], Tuple[pd.DataFrame, pd.DataFrame]
    ] = read_overall_table
    input_params_type: Type = ANNInputParams
    output_params_type: Type = ANNOutputParams


@dataclass
class DataLoaderCNN(BaseDataLoader):
    raw_data_reader: Callable[
        [os.PathLike], Tuple[pd.DataFrame, pd.DataFrame]
    ] = read_overall_table
    input_params_type: Type = CNNInputParams
    output_params_type: Type = CNNOutputParams
