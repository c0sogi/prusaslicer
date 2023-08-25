from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, get_args

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .logger import ApiLogger

InputParams = Literal["bedtemp", "exttemp", "layerthickness", "infillspeed"]

OutputParams = Literal[
    "weight", "width1", "width2", "width3", "height", "depth", "strength"
]

logger = ApiLogger(__name__)


@dataclass
class ANNConfig:
    # 기본 설정
    seed: int = 777
    print_per_epoch: int = 100
    csv_path: str = "raw_data.csv"
    metrics: List[str] = field(default_factory=lambda: ["mse", "mae", "mape"])

    # 하이퍼 파라미터
    lrs: Tuple[float, ...] = (0.001, 0.005, 0.01)  # Learning Rates
    n1s: Tuple[int, ...] = (60, 70, 80, 90, 100, 110, 120, 130)
    n2s: Tuple[int, ...] = (50, 60, 70, 80, 90, 100, 110)

    # 고정 하이퍼파라미터 : 입력/출력층 뉴런 수, 학습 Epoch 수
    dim_in: int = 50
    dim_out: int = 1
    epochs: int = 2000
    batch_size: int = 100
    kfold_splits: int = 6
    patience: int = 1000

    # 아래는 자동으로 계산됨
    number_of_cases: int = field(init=False, repr=False)
    number_of_experiments: int = field(init=False, repr=False)
    number_of_inputs: int = field(init=False, repr=False)
    number_of_outputs: int = field(init=False, repr=False)

    input_data: pd.DataFrame = field(init=False, repr=False)
    output_data: pd.DataFrame = field(init=False, repr=False)
    train_data: pd.DataFrame = field(init=False, repr=False)
    train_label: pd.DataFrame = field(init=False, repr=False)

    input_column_names: List[InputParams] = field(init=False, repr=False)
    output_column_names: List[OutputParams] = field(init=False, repr=False)

    max_input_values: Dict[InputParams, float] = field(
        default_factory=dict, repr=False
    )
    min_input_values: Dict[InputParams, float] = field(
        default_factory=dict, repr=False
    )
    max_output_values: Dict[OutputParams, float] = field(
        default_factory=dict, repr=False
    )
    min_output_values: Dict[OutputParams, float] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        tf.random.set_seed(self.seed)
        df = pd.read_csv(self.csv_path, header=None)

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
        self.input_data = self.input_data
        self.output_data = self.output_data
        x_columns: List[InputParams] = self.input_data.columns.tolist()  # type: ignore  # noqa: E501
        y_columns: List[OutputParams] = self.output_data.columns.tolist()  # type: ignore  # noqa: E501
        x_params = list(get_args(InputParams))
        y_params = list(get_args(OutputParams))
        assert isinstance(x_columns, list) and isinstance(y_columns, list)
        assert set(x_params) == set(x_columns), f"{x_columns} != {x_params}"
        assert set(y_params) == set(y_columns), f"{y_columns} != {y_params}"
        self.input_column_names = x_columns
        self.output_column_names = y_columns
        self.number_of_cases = len(self.lrs) * len(self.n1s) * len(self.n2s)

        # 최대/최소값 계산
        for data, column_names, max_values, min_values in (
            (
                self.input_data,
                self.input_column_names,
                self.max_input_values,
                self.min_input_values,
            ),
            (
                self.output_data,
                self.output_column_names,
                self.max_output_values,
                self.min_output_values,
            ),
        ):
            for column_name in column_names:
                max_values[column_name] = data[column_name].max()  # type: ignore  # noqa: E501
                min_values[column_name] = data[column_name].min()  # type: ignore  # noqa: E501
                logger.debug(
                    f"{column_name}: {min_values[column_name]} ~ {max_values[column_name]}"  # type: ignore  # noqa: E501
                )
        scaler = MinMaxScaler()
        scaler.fit(self.input_data)
        self.train_data = pd.DataFrame(
            scaler.transform(self.input_data), dtype=float
        )
        self.train_label = pd.DataFrame(
            self.get_output_data("strength"), dtype=float
        )

    def get_input_data(self, column_name: InputParams) -> pd.Series:
        return self.input_data[column_name]

    def get_output_data(self, column_name: OutputParams) -> pd.Series:
        return self.output_data[column_name]
