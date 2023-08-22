from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .logger import ApiLogger

InputParams = Literal[
    "weight", "width1", "width2", "width3", "height", "depth", "strength"
]

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
    metrics: list[str] = field(default_factory=lambda: ["mse", "mae", "mape"])

    # 하이퍼 파라미터
    lrs: tuple[float, ...] = (0.001, 0.005, 0.01)  # Learning Rates
    n1s: tuple[int, ...] = (60, 70, 80, 90, 100, 110, 120, 130)
    n2s: tuple[int, ...] = (50, 60, 70, 80, 90, 100, 110)

    # 고정 하이퍼파라미터 : 입력/출력층 뉴런 수, 학습 Epoch 수
    dim_in: int = 50
    dim_out: int = 1
    epochs: int = 2000
    batch_size: int = 100

    # 아래는 자동으로 계산됨
    number_of_cases: int = field(init=False)
    number_of_experiments: int = field(init=False)
    number_of_inputs: int = field(init=False)
    number_of_outputs: int = field(init=False)

    input_data: pd.DataFrame = field(init=False)
    output_data: pd.DataFrame = field(init=False)
    train_data: pd.DataFrame = field(init=False)
    train_label: pd.DataFrame = field(init=False)

    input_column_names: list[InputParams] = field(init=False)
    output_column_names: list[OutputParams] = field(init=False)

    max_input_values: dict[InputParams, float] = field(default_factory=dict)
    min_input_values: dict[InputParams, float] = field(default_factory=dict)
    max_output_values: dict[OutputParams, float] = field(default_factory=dict)
    min_output_values: dict[OutputParams, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tf.random.set_seed(self.seed)
        df = pd.read_csv(self.csv_path, header=None)

        # 첫 번째 행을 기반으로 X, Y 변수 구분
        x_slice = df.loc[1, df.iloc[0] == "X"].tolist()  # type: ignore
        y_slice = df.loc[1, df.iloc[0] == "Y"].tolist()  # type: ignore

        # 두 번째 행을 기반으로 열 이름 설정
        df.columns = ["Name"] + df.iloc[1].tolist()[1:]
        df = df.drop([0, 1]).reset_index(drop=True)

        # 'Name' 열의 값을 인덱스로 설정
        df.set_index("Name", inplace=True)

        # Input 및 Output DataFrame으로 분리
        self.input_data = df[x_slice]
        self.output_data = df[y_slice]

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
        self.input_column_names = self.input_data.columns.tolist()
        self.output_column_names = self.output_data.columns.tolist()
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
                max_values[column_name] = data[column_name].max()
                min_values[column_name] = data[column_name].min()
                logger.debug(
                    f"{column_name}: {min_values[column_name]} ~ {max_values[column_name]}"  # noqa: E501
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
