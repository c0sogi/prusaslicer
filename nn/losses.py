# flake8: noqa
from typing import Callable, Dict, List, Literal, Union

import tensorflow as tf
from keras import losses

LOSS_KEYS = Literal[
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
LOSS_FUNC_DICT: Dict[LOSS_KEYS, Callable] = {
    "mae": losses.MeanAbsoluteError(),
    "mse": losses.MeanSquaredError(),
    "mape": losses.MeanAbsolutePercentageError(),
    "msle": losses.MeanSquaredLogarithmicError(),
    "huber": losses.Huber(),
    "logcosh": losses.LogCosh(),
    "poisson": losses.Poisson(),
    "cosine": losses.CosineSimilarity(),
    "kld": losses.KLDivergence(),
    "binary_crossentropy": losses.BinaryCrossentropy(),
    "categorical_crossentropy": losses.CategoricalCrossentropy(),
    "sparse_categorical_crossentropy": losses.SparseCategoricalCrossentropy(),
    "hinge": losses.Hinge(),
    "squared_hinge": losses.SquaredHinge(),
    "categorical_hinge": losses.CategoricalHinge(),
    "kl_divergence": losses.KLDivergence(),
}


LossLike = Union[Callable[[tf.Tensor, tf.Tensor], tf.Tensor], LOSS_KEYS]


def convert_to_loss_func(
    loss_func: LossLike,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if isinstance(loss_func, str):
        return LOSS_FUNC_DICT[loss_func]
    if callable(loss_func):
        return loss_func
    raise ValueError(f"Unknown loss function type: {type(loss_func)}")


def weighted_loss(
    *loss_weights: float,
    loss_funcs: Union[LossLike, LOSS_KEYS, List[LossLike], List[LOSS_KEYS]],
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if isinstance(loss_funcs, (str, Callable)):
        lfs = [
            convert_to_loss_func(loss_funcs) for _ in range(len(loss_weights))
        ]
    else:  # It's a list
        lfs = [convert_to_loss_func(lf) for lf in loss_funcs]
    assert len(loss_weights) == len(
        lfs
    ), "The number of loss weights must equal the number of loss functions"
    assert all(
        callable(lf) for lf in lfs
    ), "loss_funcs must be a Callable, a list of Callables, or a list of LOSS_KEYS"
    assert len(loss_weights) == len(
        lfs
    ), "The number of loss weights must equal the number of loss functions"

    def compute_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        for i, (w, lf) in enumerate(zip(loss_weights, lfs)):
            if i == 0:
                loss = w * lf(y_true[:, i], y_pred[:, i])
            else:
                loss += w * lf(y_true[:, i], y_pred[:, i])  # type: ignore
        return loss  # type: ignore

    return compute_loss
