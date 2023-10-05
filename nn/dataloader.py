import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Tuple,
    TypedDict,
    Union,
    get_args,
)

import pandas as pd
from sklearn.model_selection import KFold

from nn.utils.logger import ApiLogger

from .config import BaseModelConfig
from .schemas import (
    ANNInputParams,
    ANNOutputParams,
    CNNInputParams,
    CNNOutputParams,
    PickleHistory,
)

PathLike = Union[str, Path]
InputParams = Union[ANNInputParams, CNNInputParams]
OutputParams = Union[ANNOutputParams, CNNOutputParams]

logger = ApiLogger(__name__)


class SSCurve(TypedDict):
    strain: List[float]
    stress: List[float]


class SSData(TypedDict):
    idx: int
    curve: SSCurve


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


def read_all_ss_curves(
    csv_directory_path: os.PathLike,
) -> Dict[int, List[SSData]]:
    ss_data_dict: Dict[int, List[SSData]] = {}
    for csv_file_path in Path(csv_directory_path).glob("*.csv"):
        seperated: List[str] = csv_file_path.stem.split("_")
        _case: int = int(seperated[0])
        _idx: int = int(seperated[1])
        assert len(seperated) == 2, f"{csv_file_path} is not valid"

        df = read_single_ss_curve(csv_file_path)
        try:
            strain = df["변형율"].tolist()
            stress = df["강도"].tolist()
            data = SSData(
                idx=_idx, curve=SSCurve(strain=strain, stress=stress)
            )
        except KeyError:
            logger.error(f"{csv_file_path} is not valid")
            continue

        if _case not in ss_data_dict:
            ss_data_dict[_case] = [data]
        else:
            ss_data_dict[_case].append(data)

    # Sort
    for _case in ss_data_dict.keys():
        ss_data_dict[_case].sort(key=lambda x: x["idx"])
    ss_data_dict = dict(sorted(ss_data_dict.items()))
    return ss_data_dict


@dataclass
class BaseDataLoader(ABC):
    model_config: BaseModelConfig
    train_data: pd.DataFrame = field(init=False, repr=False)  # to be filled
    train_label: pd.DataFrame = field(init=False, repr=False)  # to be filled

    @abstractmethod
    def dataset_batch_iterator(
        self, batch_size: int
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        ...

    @abstractmethod
    def dataset_kfold_iterator(
        self, n_splits: int
    ) -> Iterator[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ]:
        ...

    @abstractmethod
    def get_input_data(self, column_name: InputParams) -> pd.Series:
        ...

    @abstractmethod
    def get_output_data(self, column_name: OutputParams) -> pd.Series:
        ...


@dataclass
class DataLoaderANN(BaseDataLoader):
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
        x_columns: List[ANNInputParams] = self.input_data.columns.tolist()  # type: ignore  # noqa: E501
        y_columns: List[ANNOutputParams] = self.output_data.columns.tolist()  # type: ignore  # noqa: E501
        x_params = list(get_args(ANNInputParams))
        y_params = list(get_args(ANNOutputParams))
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
        self.train_label = pd.concat(
            [
                self.get_output_data("strength"),
                self.get_output_data("lengthavg"),
            ],
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

    def get_input_data(self, column_name: ANNInputParams) -> pd.Series:
        return self.input_data[column_name]

    def get_output_data(self, column_name: ANNOutputParams) -> pd.Series:
        return self.output_data[column_name]


@dataclass
class DataLoaderCNN(BaseDataLoader):
    def __post_init__(self):
        df = pd.read_csv(
            self.model_config.input_path, header=None, encoding="cp949"
        )

        x_indices: pd.Index = df.columns[df.iloc[0] == "변형율"] - 1
        y_indices: pd.Index = df.columns[df.iloc[0] == "강도"] - 1

        # 두 번째 행을 기반으로 열 이름 설정
        df.columns = ["Name"] + df.iloc[0].tolist()[1:]
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
        x_columns: List[CNNInputParams] = self.input_data.columns.tolist()  # type: ignore  # noqa: E501
        y_columns: List[CNNOutputParams] = self.output_data.columns.tolist()  # type: ignore  # noqa: E501
        x_params = list(get_args(CNNInputParams))
        y_params = list(get_args(CNNOutputParams))
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        assert set(x_params) == set(x_columns), f"{x_columns} != {x_params}"
        assert set(y_params) == set(y_columns), f"{y_columns} != {y_params}"
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        self.input_column_names = x_columns
        self.output_column_names = y_columns
        self.train_data = pd.concat(
            [
                self.get_input_data(column_name)
                for column_name in self.input_column_names
            ],
            axis=1,
        ).astype(float)
        self.train_label = pd.concat(
            [
                self.get_output_data("강도"),
            ],
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

    def get_input_data(self, column_name: CNNInputParams) -> pd.Series:
        return self.input_data[column_name]

    def get_output_data(self, column_name: CNNOutputParams) -> pd.Series:
        return self.output_data[column_name]
