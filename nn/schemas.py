from typing import Dict, Iterable, List, Literal, TypedDict, Union, get_args

from typing_extensions import NotRequired

from .config import BaseModelConfig


HyperParamValue = Union[int, float]
HyperParamsDict = Dict[str, HyperParamValue]
HyperParamsDictAll = Dict[str, Iterable[HyperParamValue]]


ANNInputParams = Literal[
    "bedtemp",
    "exttemp",
    "layerthickness",
    "infillspeed",
    "density",
    "thermalresistance",
    "impactstrength",
    "glasstransitiontemp",
    "thermalconductivity",
    "linearthermalexpansioncoefficient",
]

ANNOutputParams = Literal[
    # "weight",
    # "width1",
    # "width2",
    # "width3",
    # "height",
    # "depth",
    "strength",
    "lengthavg",
]


_ANN_INPUT_PARAM_ARGS = get_args(ANNInputParams)
ANN_INPUT_PARAM_INDICES = (
    _ANN_INPUT_PARAM_ARGS.index("bedtemp"),
    _ANN_INPUT_PARAM_ARGS.index("exttemp"),
    _ANN_INPUT_PARAM_ARGS.index("layerthickness"),
    _ANN_INPUT_PARAM_ARGS.index("infillspeed"),
    _ANN_INPUT_PARAM_ARGS.index("density"),
    _ANN_INPUT_PARAM_ARGS.index("thermalresistance"),
    _ANN_INPUT_PARAM_ARGS.index("impactstrength"),
    _ANN_INPUT_PARAM_ARGS.index("glasstransitiontemp"),
    _ANN_INPUT_PARAM_ARGS.index("thermalconductivity"),
    _ANN_INPUT_PARAM_ARGS.index("linearthermalexpansioncoefficient"),
)

CNNInputParams = Literal["변형율"]
CNNOutputParams = Literal["강도"]


class TrainInput(TypedDict):
    hyper_params: HyperParamsDict
    config: BaseModelConfig


class TrainOutput(TypedDict):
    loss: NotRequired[List[float]]
    mae: NotRequired[List[float]]
    mape: NotRequired[List[float]]
    val_loss: NotRequired[List[float]]
    val_mse: NotRequired[List[float]]
    val_mae: NotRequired[List[float]]
    val_mape: NotRequired[List[float]]
    rmse: NotRequired[List[float]]


class PickleHistory(TypedDict):
    train_input: TrainInput
    train_output: TrainOutput
