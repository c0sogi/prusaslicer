import itertools
import json
import multiprocessing
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Iterable, Optional, Type, TypedDict

import numpy as np
from tensorflow import keras

from .callbacks import AccuracyPerEpoch
from .config import ANNConfig
from .dataloader import dataset_kfold_iterator, dump_pickle
from .logger import ApiLogger

logger = ApiLogger(__name__)


class TrainInput(TypedDict):
    case: int
    hyper_params: dict[str, int | float]
    config: ANNConfig


class PickleHistory(TypedDict):
    train_input: TrainInput
    train_output: list | dict


def process_history(
    hist_history: dict[str, list[float]]
) -> dict[str, list[float]]:
    if "mse" in hist_history:
        hist_history["rmse"] = np.sqrt(hist_history["mse"]).tolist()
        hist_history.pop("mse")
    return hist_history


@dataclass
class Trainer:
    model_class: Type[keras.Sequential]
    model_config: ANNConfig
    model_name: Optional[str] = None
    workers: int = multiprocessing.cpu_count()
    use_multiprocessing: bool = True

    def __post_init__(self) -> None:
        self._model_name = self.model_name or str(self.model_class.__name__)

    def train(
        self, case: int, hyper_params: Optional[dict[str, int | float]] = None
    ) -> PickleHistory:
        # Resets all state generated by Keras.
        hyper_params = hyper_params or {}
        model_config = self.model_config
        keras.backend.clear_session()
        callback = AccuracyPerEpoch(
            print_per_epoch=model_config.print_per_epoch
        )
        model = self.model_class(
            model_config,
            **{
                key: value
                for key, value in hyper_params.items()
                if key in signature(self.model_class.__init__).parameters
            },
        )
        # model_train = model.fit(
        #     verbose=0,  # type: ignore
        #     callbacks=[callback],
        #     batch_size=model_config.batch_size,
        #     use_multiprocessing=self.use_multiprocessing,
        #     validation_data=(x_test, y_test),
        #     workers=self.workers,
        # )

        pickle_history: PickleHistory = PickleHistory(
            train_input=TrainInput(
                case=case, hyper_params=hyper_params, config=model_config
            ),
            train_output={},
        )
        kfold_splits = model_config.kfold_splits
        Path("./output").mkdir(exist_ok=True, parents=True)

        logger.info(f"Start training: {pickle_history}")
        if kfold_splits > 0:
            histories: list = []
            redundant = model_config.epochs % kfold_splits  # type: int
            for kfold_case, (x_train, y_train, x_test, y_test) in enumerate(
                dataset_kfold_iterator(
                    model_config.train_data,
                    model_config.train_label,
                    kfold_splits,
                ),
                start=1,
            ):
                logger.info(
                    f"Kfolds: {kfold_case}/{model_config.kfold_splits}"
                )
                hist = model.fit(
                    x_train,
                    y_train,
                    epochs=model_config.epochs // kfold_splits + redundant,
                    verbose=0,  # type: ignore
                    callbacks=[callback],
                    batch_size=model_config.batch_size,
                    use_multiprocessing=self.use_multiprocessing,
                    validation_data=(x_test, y_test),
                    workers=self.workers,
                )
                redundant = 0
                histories.append(process_history(hist.history))
            mean_last_history = {
                key: np.mean(histories[-1][key], axis=0)
                for key in histories[0].keys()
            }
            pickle_history["train_output"] = histories
        else:
            hist = model.fit(
                model_config.train_data,
                model_config.train_label,
                epochs=model_config.epochs,
                verbose=0,  # type: ignore
                callbacks=[callback],
                batch_size=model_config.batch_size,
                use_multiprocessing=self.use_multiprocessing,
                workers=self.workers,
            )
            last_history = process_history(hist.history)
            mean_last_history = {
                key: str(np.mean(last_history[key], axis=0))
                for key in last_history.keys()
            }
            pickle_history["train_output"] = last_history
        model.save(self.get_checkpoint_filename(case))
        logger.info(f"End training: {json.dumps(mean_last_history, indent=2)}")
        return pickle_history

    def hyper_train(
        self, hyper_params: Optional[dict[str, Iterable[int | float]]] = None
    ) -> None:
        hyper_params = hyper_params or {}
        product = tuple(itertools.product(*hyper_params.values()))
        logger.critical(
            f"model: {self._model_name} with {self.model_config.number_of_cases} cases"  # noqa: E501
        )
        pickled_histories = []  # type: list[PickleHistory]
        for case, combined_hyper_params in enumerate(product, start=1):
            pickled_histories.append(
                self.train(
                    case,
                    hyper_params=dict(
                        zip(hyper_params, combined_hyper_params)
                    ),
                )
            )
        dump_pickle(self.get_result_filename(), pickled_histories)

    def get_checkpoint_filename(self, case: int) -> str:
        model_config = self.model_config
        model_name = self._model_name
        if model_config.kfold_splits <= 0:
            return f"./output/{model_name}_E{model_config.epochs}_C{case}of{model_config.number_of_cases}.keras"  # noqa: E501
        return f"./output/{model_name}_E{model_config.epochs}_C{case}of{model_config.number_of_cases}_K{model_config.kfold_splits}.keras"  # noqa: E501

    def get_result_filename(self) -> str:
        model_config = self.model_config
        model_name = self._model_name
        if model_config.kfold_splits <= 0:
            return f"./output/{model_name}_E{model_config.epochs}_C{model_config.number_of_cases}.pickle"  # noqa: E501
        return f"./output/{model_name}_E{model_config.epochs}_C{model_config.number_of_cases}_K{model_config.kfold_splits}.pickle"  # noqa: E501
