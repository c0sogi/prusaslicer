import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

from .config import BaseModelConfig
from .typings import PickleHistory, SSCurve
from .utils.logger import ApiLogger

logger = ApiLogger(__name__)


def _normalize_1d_sequence(sequence: np.ndarray, trg_len: int) -> np.ndarray:
    assert len(sequence.shape) == 1, sequence.shape
    src_len = len(sequence)
    f = interp1d(
        np.linspace(0, src_len - 1, src_len),
        sequence,
        kind="linear",
    )
    seq_new = f(np.linspace(0, src_len - 1, trg_len))
    return (seq_new - np.min(seq_new)) / (np.max(seq_new) - np.min(seq_new))


def _normalize_2d_sequence(matrix: np.ndarray, trg_len: int) -> np.ndarray:
    assert len(matrix.shape) == 2, matrix.shape
    return np.array([_normalize_1d_sequence(row, trg_len) for row in matrix])


def _read_single_ss_curve(
    csv_file_path: os.PathLike,
) -> pd.DataFrame:
    assert Path(csv_file_path).exists(), f"{csv_file_path} does not exist"
    df = pd.read_csv(csv_file_path, header=None, encoding="cp949")

    # 두 번째 행을 기반으로 열 이름 설정
    df.columns = ["Name"] + df.iloc[0].tolist()[1:]
    df = df.drop([0, 1]).reset_index(drop=True)
    return df


def _read_ss_curves(
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

            df = _read_single_ss_curve(csv_file_path)
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
        df.set_index("Name", inplace=True)
        frames.append(df)
    return pd.concat(frames)


def _read_x_and_y_from_table(
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


def read_all(
    raw_data_dir: os.PathLike = Path("./raw_data"),
    table_filename: str = "table.csv",
    drop_invalid: bool = True,
) -> Dict[Literal["x_lstm", "x_ann", "y_ann"], pd.DataFrame]:
    # Load raw data (ss curves and table)
    x_lstm = _read_ss_curves(raw_data_dir)
    x_ann, y_ann = _read_x_and_y_from_table(
        Path(raw_data_dir) / table_filename
    )

    # Filter out invalid data from x_ann and y_ann
    keys = set(x_ann.index) & set(y_ann.index) & set(x_lstm.index)
    logger.debug(f"===== Number of valid data: {len(keys)} =====")
    if drop_invalid:
        x_ann = x_ann[x_ann.index.isin(keys)]
        y_ann = y_ann[y_ann.index.isin(keys)]
        x_lstm = x_lstm[x_lstm.index.isin(keys)]
    _shape = x_lstm.shape

    # Merge x_lstm, x_ann, and y_ann
    for to_merge in (x_ann, y_ann):
        x_lstm = x_lstm.merge(to_merge, on="Name", how="left")
    x_lstm.dropna(inplace=True)
    assert _shape[0] == x_lstm.shape[0], f"{_shape} != {x_lstm.shape}"
    return {
        "x_lstm": x_lstm,
        "x_ann": x_ann,
        "y_ann": y_ann,
    }


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

    input_dataset: pd.DataFrame
    output_dataset: pd.DataFrame

    train_input_params: Set[str]
    train_output_params: Set[str]

    train_inputs: pd.DataFrame = field(init=False, repr=False)
    train_outputs: pd.DataFrame = field(init=False, repr=False)

    # To be filled

    def __post_init__(self):
        assert (
            self.input_dataset.shape[0] == self.output_dataset.shape[0]
        ), f"{self.input_dataset.shape} != {self.output_dataset.shape}"

        x_columns: List[str] = self.input_dataset.columns.tolist()
        y_columns: List[str] = self.output_dataset.columns.tolist()
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        assert self.train_input_params == set(
            x_columns
        ), f"{x_columns} != {self.train_input_params}"
        assert self.train_output_params == set(
            y_columns
        ), f"{y_columns} != {self.train_output_params}"

        self.train_inputs = pd.concat(
            [self.get_input_data(column_name) for column_name in x_columns],
            axis=1,
        ).astype(float)
        self.train_outputs = pd.concat(
            [self.get_output_data(column_name) for column_name in y_columns],
            axis=1,
        ).astype(float)

        logger.debug(f"===== Train inputs: {self.train_inputs.shape} =====")
        logger.debug(self.train_inputs.head(48))
        logger.debug(
            f"===== Train outputs: {self.train_outputs.shape} ====="
        )
        logger.debug(self.train_outputs.head(3))

    def dataset_batch_iterator(
        self, batch_size: int = 1
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        x_data, y_data = self.train_inputs, self.train_outputs
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
        x_data, y_data = self.train_inputs, self.train_outputs
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
        return self.input_dataset[column_name]

    def get_output_data(self, column_name: str) -> pd.Series:
        return self.output_dataset[column_name]
