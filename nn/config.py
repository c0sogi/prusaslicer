from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, get_args

from keras.src import activations

from .typings import LossFuncsString, LossKeys
from .utils.logger import ApiLogger

logger = ApiLogger(__name__)


@dataclass
class BaseModelConfig:
    # 기본 설정
    seed: int = 777
    print_per_epoch: int = 100
    output_path: str = "./output"
    metrics: List[str] = field(
        default_factory=lambda: ["mse", "mae", "mape"]
    )
    epochs: int = 2000
    batch_size: int = 100
    kfold_splits: int = 6
    patience: int = 1000
    dim_out: int = 2

    # 하이퍼파라미터
    lr: float = 0.001
    loss_funcs: LossFuncsString = field(
        default_factory=lambda: ["mae", "mae"]
    )
    loss_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    activation: str = "relu"

    # Overfitting 방지
    l1_reg: Optional[float] = None
    l2_reg: Optional[float] = None
    dropout_rate: float = 0.0
    normalize_layer: bool = False

    def __post_init__(self) -> None:
        try:
            activations.get(self.activation)
        except Exception as e:
            raise ValueError(
                f"{self.activation}은 잘못된 Activation Function입니다: {e}"
            )

        assert self.dim_out > 0, "출력층 뉴런 수는 0보다 커야 합니다."
        assert self.epochs > 0, "학습 Epoch 수는 0보다 커야 합니다."
        assert self.batch_size > 0, "Batch Size는 0보다 커야 합니다."
        assert self.kfold_splits >= 0, "K-Fold Splits는 0보다 크거나 같아야 합니다."
        assert self.patience > 0, "Patience는 0보다 커야 합니다."
        if isinstance(self.loss_funcs, str):
            assert self.loss_funcs in get_args(
                LossKeys
            ), "잘못된 Loss Function입니다."
        else:
            assert all(
                lf in get_args(LossKeys) for lf in self.loss_funcs
            ), "잘못된 Loss Function입니다."
        assert all(
            0.0 <= w <= 1.0 for w in self.loss_weights
        ), "Loss Weights는 0과 1 사이의 값이어야 합니다."
        self.loss_weights = [
            w / sum(self.loss_weights) for w in self.loss_weights
        ]  # Loss weights의 합이 1이 되도록 정규화
        assert (
            len(self.loss_weights) == self.dim_out
        ), "Loss Weights의 길이는 출력층 뉴런 수와 같아야 합니다."

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(
            **{
                k: v
                for k, v in config_dict.items()
                if k in {f.name for f in fields(cls)}
            }
        )


@dataclass
class ANNModelConfig(BaseModelConfig):
    n1: int = 60
    n2: int = 50
    n3: int = 50


@dataclass
class LSTMModelConfig(BaseModelConfig):
    seq_len: int = -1
    lstm_units: int = -1

    # # 아래는 자동으로 계산됨
    # number_of_experiments: int = field(init=False, repr=False)
    # number_of_inputs: int = field(init=False, repr=False)
    # number_of_outputs: int = field(init=False, repr=False)
    # input_data: pd.DataFrame = field(init=False, repr=False)
    # output_data: pd.DataFrame = field(init=False, repr=False)
    # train_data: pd.DataFrame = field(init=False, repr=False)
    # train_label: pd.DataFrame = field(init=False, repr=False)
    # input_column_names: List[InputParams] = field(init=False, repr=False)
    # output_column_names: List[OutputParams] = field(init=False, repr=False)
    # max_input_values: Dict[InputParams, float] = field(
    #     default_factory=dict, repr=False, init=False
    # )
    # min_input_values: Dict[InputParams, float] = field(
    #     default_factory=dict, repr=False, init=False
    # )
    # max_output_values: Dict[OutputParams, float] = field(
    #     default_factory=dict, repr=False, init=False
    # )
    # min_output_values: Dict[OutputParams, float] = field(
    #     default_factory=dict, repr=False, init=False
    # )

    # def __post_init__(self) -> None:
    #     df = pd.read_csv(self.input_path, header=None)

    #     x_indices: pd.Index = df.columns[df.iloc[0] == "X"] - 1
    #     y_indices: pd.Index = df.columns[df.iloc[0] == "Y"] - 1

    #     # 두 번째 행을 기반으로 열 이름 설정
    #     df.columns = ["Name"] + df.iloc[1].tolist()[1:]
    #     df = df.drop([0, 1]).reset_index(drop=True)

    #     # 'Name' 열의 값을 인덱스로 설정
    #     df.set_index("Name", inplace=True)

    #     # Input 및 Output DataFrame으로 분리
    #     self.input_data = df.iloc[:, x_indices]
    #     self.output_data = df.iloc[:, y_indices]

    #     logger.debug(
    #         f"===== Input Data: {self.input_data.shape} =====\n{self.input_data.head(3)}"  # noqa: E501
    #     )
    #     logger.debug(
    #         f"===== Output Data: {self.output_data.shape} =====\n{self.output_data.head(3)}"  # noqa: E501
    #     )

    #     assert (
    #         self.input_data.shape[0] == self.output_data.shape[0]
    #     ), "데이터 개수 불일치"
    #     self.number_of_experiments = self.input_data.shape[0]
    #     self.number_of_inputs = self.input_data.shape[1]
    #     self.number_of_outputs = self.output_data.shape[1]
    #     x_columns: List[InputParams] = self.input_data.columns.tolist()  # type: ignore  # noqa: E501
    #     y_columns: List[OutputParams] = self.output_data.columns.tolist()  # type: ignore  # noqa: E501
    #     x_params = list(get_args(InputParams))
    #     y_params = list(get_args(OutputParams))
    #     assert isinstance(x_columns, list) and isinstance(y_columns, list)
    #     assert set(x_params) == set(x_columns), f"{x_columns} != {x_params}"
    #     assert set(y_params) == set(y_columns), f"{y_columns} != {y_params}"
    #     self.input_column_names = x_columns
    #     self.output_column_names = y_columns

    #     # # 최대/최소값 계산
    #     # for data, column_names, max_values, min_values in (
    #     #     (
    #     #         self.input_data,
    #     #         self.input_column_names,
    #     #         self.max_input_values,
    #     #         self.min_input_values,
    #     #     ),
    #     #     (
    #     #         self.output_data,
    #     #         self.output_column_names,
    #     #         self.max_output_values,
    #     #         self.min_output_values,
    #     #     ),
    #     # ):
    #     #     for column_name in column_names:
    #     #         max_values[column_name] = data[column_name].max()  # type: ignore  # noqa: E501
    #     #         min_values[column_name] = data[column_name].min()  # type: ignore  # noqa: E501
    #     #         logger.debug(
    #     #             f"{column_name}: {min_values[column_name]} ~ {max_values[column_name]}"  # type: ignore  # noqa: E501
    #     #         )
    #     # scaler = MinMaxScaler()
    #     # scaler.fit(self.input_data)
    #     # self.train_data = pd.DataFrame(
    #     #     scaler.transform(self.input_data), dtype=float
    #     # )
    #     self.train_data = pd.concat(
    #         [
    #             self.get_input_data(column_name)
    #             for column_name in self.input_column_names
    #         ],
    #         axis=1,
    #     ).astype(float)
    #     self.train_label = pd.DataFrame(
    #         self.get_output_data("strength"), dtype=float
    #     )
    #     logger.debug(f"===== Train Data: {self.train_data.shape} =====")
    #     logger.debug(self.train_data.head(48))
    #     logger.debug(f"===== Train Label: {self.train_label.shape} =====")
    #     logger.debug(self.train_label.head(3))

    # def get_input_data(self, column_name: InputParams) -> pd.Series:
    #     return self.input_data[column_name]

    # def get_output_data(self, column_name: OutputParams) -> pd.Series:
    #     return self.output_data[column_name]
