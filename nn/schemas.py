from typing import Dict, Iterable, List, Literal, TypedDict, Union, get_args

from typing_extensions import NotRequired

from .config import ModelConfig


HyperParamValue = Union[int, float]
HyperParamsDict = Dict[str, HyperParamValue]
HyperParamsDictAll = Dict[str, Iterable[HyperParamValue]]


InputParams = Literal[
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

OutputParams = Literal[
    "weight", "width1", "width2", "width3", "height", "depth", "strength"
]


INPUT_PARAM_ARGS = get_args(InputParams)
INPUT_PARAM_INDICES = (
    INPUT_PARAM_ARGS.index("bedtemp"),
    INPUT_PARAM_ARGS.index("exttemp"),
    INPUT_PARAM_ARGS.index("layerthickness"),
    INPUT_PARAM_ARGS.index("infillspeed"),
    INPUT_PARAM_ARGS.index("density"),
    INPUT_PARAM_ARGS.index("thermalresistance"),
    INPUT_PARAM_ARGS.index("impactstrength"),
    INPUT_PARAM_ARGS.index("glasstransitiontemp"),
    INPUT_PARAM_ARGS.index("thermalconductivity"),
    INPUT_PARAM_ARGS.index("linearthermalexpansioncoefficient"),
)


class TrainInput(TypedDict):
    hyper_params: HyperParamsDict
    config: ModelConfig


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
