from typing import Dict, Iterable, List, TypedDict, Union

from typing_extensions import NotRequired

from .config import ModelConfig


HyperParamValue = Union[int, float]
HyperParamsDict = Dict[str, HyperParamValue]
HyperParamsDictAll = Dict[str, Iterable[HyperParamValue]]


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
