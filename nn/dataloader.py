import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, get_args

import pandas as pd
from sklearn.model_selection import KFold

from .config import ModelConfig
from .schemas import PickleHistory, InputParams, OutputParams
from .utils.logger import ApiLogger


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
    Path(file_path).write_text("\n".join(json.dumps(entry) for entry in data))


def load_jsonl(file_path: PathLike) -> List[Dict[str, object]]:
    return [
        json.loads(line) for line in Path(file_path).read_text().splitlines()
    ]


@dataclass
class DataLoader:
    model_config: ModelConfig

    def __post_init__(self):
        df = pd.read_csv(self.model_config.input_path, header=None)

        x_indices: pd.Index = df.columns[df.iloc[0] == "X"] - 1
        y_indices: pd.Index = df.columns[df.iloc[0] == "Y"] - 1

        # 두 번째 행을 기반으로 열 이름 설정
        df.columns = ["Name"] + df.iloc[1].tolist()[1:]
        df = df.drop([0, 1]).reset_index(drop=True)

        # 'Name' 열의 값을 인덱스로 설정
        df.set_index("Name", inplace=True)

        # Input 및 Output DataFrame으로 분리
        self.input_data = df.iloc[:, x_indices]
        self.output_data = df.iloc[:, y_indices]

        logger.debug(
            f"===== Input Data: {self.input_data.shape} =====\n{self.input_data.head(3)}"  # noqa: E501
        )
        logger.debug(
            f"===== Output Data: {self.output_data.shape} =====\n{self.output_data.head(3)}"  # noqa: E501
        )

        assert (
            self.input_data.shape[0] == self.output_data.shape[0]
        ), "데이터 개수 불일치"
        self.number_of_experiments = self.input_data.shape[0]
        self.number_of_inputs = self.input_data.shape[1]
        self.number_of_outputs = self.output_data.shape[1]
        x_columns: List[InputParams] = self.input_data.columns.tolist()  # type: ignore  # noqa: E501
        y_columns: List[OutputParams] = self.output_data.columns.tolist()  # type: ignore  # noqa: E501
        x_params = list(get_args(InputParams))
        y_params = list(get_args(OutputParams))
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        assert set(x_params) == set(x_columns), f"{x_columns} != {x_params}"
        assert set(y_params) == set(y_columns), f"{y_columns} != {y_params}"
        self.input_column_names = x_columns
        self.output_column_names = y_columns
        self.train_data = pd.concat(
            [
                self.get_input_data(column_name)
                for column_name in self.input_column_names
            ],
            axis=1,
        ).astype(float)
        self.train_label = pd.DataFrame(
            self.get_output_data("strength"), dtype=float
        )
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
            yield x_data[batch_start:batch_end], y_data[batch_start:batch_end]

    def dataset_kfold_iterator(
        self, n_splits: int = 5
    ) -> Iterator[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ]:
        kf = KFold(n_splits=n_splits)
        x_data, y_data = self.train_data, self.train_label
        for train_index, test_index in kf.split(x_data, y_data):
            x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
            y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
            yield x_train, y_train, x_test, y_test

    def get_input_data(self, column_name: InputParams) -> pd.Series:
        return self.input_data[column_name]

    def get_output_data(self, column_name: OutputParams) -> pd.Series:
        return self.output_data[column_name]
