from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Literal,
    Mapping,
    NotRequired,
    TypedDict,
    Union,
)
import numpy as np
import pandas as pd

import tensorflow as tf

if TYPE_CHECKING:
    from .config import BaseModelConfig

SingleData = Union[pd.DataFrame, np.ndarray]
ListData = Union[List[pd.DataFrame], List[np.ndarray]]
DataLike = Union[SingleData, ListData]

HyperParamsDict = Mapping[str, Union[int, float]]
HyperParamsDictAll = Mapping[str, Union[Iterable[int], Iterable[float]]]


LossKeys = Literal[
    "mae",
    "mse",
    "mape",
    "msle",
    "huber",
    "logcosh",
    "poisson",
    "cosine",
    "kld",
    "binary_crossentropy",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
    "hinge",
    "squared_hinge",
    "categorical_hinge",
    "kl_divergence",
]

LossLike = Union[Callable[[tf.Tensor, tf.Tensor], tf.Tensor], LossKeys]
LossFuncs = Union[LossLike, LossKeys, List[LossLike], List[LossKeys]]
LossFuncsString = Union[LossKeys, List[LossKeys]]


class TrainInput(TypedDict):
    hyper_params: HyperParamsDict
    config: "BaseModelConfig"


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


class SSCurve(TypedDict):
    strain: List[float]
    stress: List[float]


class SSData(TypedDict):
    idx: int
    curve: SSCurve
