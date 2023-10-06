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

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

from .config import BaseModelConfig
from .schemas import (
    ANNInputParams,
    ANNOutputParams,
    LSTMInputParams,
    LSTMOutputParams,
)
from .typings import PickleHistory, SSCurve
from .utils.logger import ApiLogger

logger = ApiLogger(__name__)


def normalize_1d_sequence(sequence: np.ndarray, trg_len: int) -> np.ndarray:
    assert len(sequence.shape) == 1, sequence.shape
    src_len = len(sequence)
    f = interp1d(
        np.linspace(0, src_len - 1, src_len),
        sequence,
        kind="linear",
    )
    seq_new = f(np.linspace(0, src_len - 1, trg_len))
    return (seq_new - np.min(seq_new)) / (np.max(seq_new) - np.min(seq_new))


def normalize_2d_sequence(matrix: np.ndarray, trg_len: int) -> np.ndarray:
    assert len(matrix.shape) == 2, matrix.shape
    return np.array([normalize_1d_sequence(row, trg_len) for row in matrix])


def read_single_ss_curve(
    csv_file_path: os.PathLike,
) -> pd.DataFrame:
    assert Path(csv_file_path).exists(), f"{csv_file_path} does not exist"
    df = pd.read_csv(csv_file_path, header=None, encoding="cp949")

    # 두 번째 행을 기반으로 열 이름 설정
    df.columns = ["Name"] + df.iloc[0].tolist()[1:]
    df = df.drop([0, 1]).reset_index(drop=True)
    df.set_index("Name", inplace=True)

    return df


def read_ss_curves(
    raw_data_path: os.PathLike,
) -> pd.DataFrame:
    ss_data_dict: Dict[str, SSCurve] = {}
    for csv_dir_path in Path(raw_data_path).iterdir():
        for csv_file_path in csv_dir_path.glob("*.csv"):
            seperated: List[str] = csv_file_path.stem.split("_")
            try:
                key = f"{csv_dir_path.name.upper()}-{int(seperated[0])}-{int(seperated[1])}"  # noqa: E501
                assert len(seperated) == 2, f"{csv_file_path} is not valid"
            except Exception as e:
                logger.error(f"{csv_file_path} is not valid: {e}")
                continue

            df = read_single_ss_curve(csv_file_path)
            try:
                strain = df["변형율"].tolist()
                stress = df["강도"].tolist()
                ss_data_dict[key] = SSCurve(strain=strain, stress=stress)
            except KeyError:
                logger.error(f"{csv_file_path} is not valid")
                continue
    frames = []  # type: List[pd.DataFrame]
    for sample_name, curve in ss_data_dict.items():
        df = pd.DataFrame(curve)
        df["Name"] = sample_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def read_x_and_y_from_table(
    csv_file_path: os.PathLike,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_file_path, header=None)

    x_indices: pd.Index = df.columns[df.iloc[0] == "X"] - 1
    y_indices: pd.Index = df.columns[df.iloc[0] == "Y"] - 1

    # 두 번째 행을 기반으로 열 이름 설정
    df.columns = ["Name"] + df.iloc[1].tolist()[1:]
    df = df.drop([0, 1]).reset_index(drop=True)

    # 'Name' 열의 값을 인덱스로 설정
    df.set_index("Name", inplace=True)
    x = df.iloc[:, x_indices]
    y = df.iloc[:, y_indices]
    assert x.shape[0] == y.shape[0], f"{x.shape} != {y.shape}"
    return x, y


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
class BaseDataLoader:
    model_config: BaseModelConfig

    # To be filled
    train_data: pd.DataFrame = field(init=False, repr=False)
    input_params_type: Type = field(init=False, repr=False)
    output_params_type: Type = field(init=False, repr=False)
    train_label: pd.DataFrame = field(init=False, repr=False)
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

        x_params = list(get_args(self.input_params_type))
        x_columns: List[str] = self.input_data.columns.tolist()
        y_params = list(get_args(self.output_params_type))
        y_columns: List[str] = self.output_data.columns.tolist()
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        assert set(x_params) == set(x_columns), f"{x_columns} != {x_params}"
        assert set(y_params) == set(y_columns), f"{y_columns} != {y_params}"

        self.train_data = pd.concat(
            [self.get_input_data(column_name) for column_name in x_columns],
            axis=1,
        ).astype(float)
        self.train_label = pd.concat(
            [self.get_output_data(column_name) for column_name in y_columns],
            axis=1,
        ).astype(float)

        logger.debug(
            f"===== Input Data: {self.input_data.shape} =====\n{self.input_data.head(3)}"  # noqa: E501
        )
        logger.debug(
            f"===== Output Data: {self.output_data.shape} =====\n{self.output_data.head(3)}"  # noqa: E501
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
    ] = read_x_and_y_from_table
    input_params_type: Type = ANNInputParams
    output_params_type: Type = ANNOutputParams


@dataclass
class DataLoaderLSTM(BaseDataLoader):
    raw_data_reader: Callable[
        [os.PathLike], Tuple[pd.DataFrame, pd.DataFrame]
    ] = read_ss_curves
    input_params_type: Type = LSTMInputParams
    output_params_type: Type = LSTMOutputParams
