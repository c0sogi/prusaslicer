import itertools
import json
import multiprocessing
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .callbacks import AccuracyPerEpoch, EarlyStopping
from .config import BaseModelConfig
from .dataloader import (
    DataLoader,
    PickleHistory,
    dump_pickle,
    load_pickle,
)
from .typings import (
    DataLike,
    HyperParamsDict,
    HyperParamsDictAll,
    SingleData,
    TrainInput,
    TrainOutput,
)
from .utils.logger import ApiLogger

logger = ApiLogger(__name__)


@dataclass
class Trainer:
    data_loader: DataLoader
    model_class: Type[keras.Model]
    model_config: BaseModelConfig
    model_name: Optional[str] = None
    workers: int = multiprocessing.cpu_count()
    use_multiprocessing: bool = True

    def __post_init__(self) -> None:
        self._model_name = self.model_name or str(self.model_class.__name__)

    @property
    def train_inputs(self) -> DataLike:
        return self.data_loader.train_inputs

    @property
    def train_outputs(self) -> SingleData:
        return self.data_loader.train_outputs

    @property
    def kfold_splits(self) -> int:
        return self.model_config.kfold_splits

    @property
    def output_path(self) -> str:
        return self.model_config.output_path

    def train(
        self, hyper_params: Optional[HyperParamsDict] = None
    ) -> Union[Tuple[str, PickleHistory], List[Tuple[str, PickleHistory]],]:
        """Train the model with given hyper parameters."""
        # Resets all state generated by Keras.
        kfold_splits = self.kfold_splits
        if kfold_splits > 1:
            # Kfolds
            zips = []  # type: List[Tuple[str, PickleHistory]]
            for kfold_case, (x_train, y_train, x_test, y_test) in enumerate(
                self.data_loader.dataset_kfold_iterator(kfold_splits),
                start=1,
            ):
                logger.info(f"Kfolds: {kfold_case}/{kfold_splits}")
                train_file_stem, pickle_history = self._train(
                    kfold_case=kfold_case,
                    x_train=x_train,
                    y_train=y_train,
                    validation_data=(x_test, y_test),
                    hyper_params=hyper_params,
                )
                zips.append((train_file_stem, pickle_history))
            return zips
        else:
            # Normal
            return self._train(
                x_train=self.train_inputs,
                y_train=self.train_outputs,
                hyper_params=hyper_params,
            )

    def _train(
        self,
        x_train: DataLike,
        y_train: SingleData,
        validation_data: Optional[Tuple[DataLike, DataLike]] = None,
        hyper_params: Optional[HyperParamsDict] = None,
        kfold_case: Optional[int] = None,
        val_split: Optional[float] = 0.1,
    ) -> Tuple[str, PickleHistory]:
        model_config = self.apply_hyper_params(hyper_params)
        tf.random.set_seed(model_config.seed)
        keras.backend.clear_session()
        model, pickle_history = self.create_model_and_history(
            model_config, hyper_params, kfold_case
        )
        if self.get_current_epoch(pickle_history) > 0:
            logger.info("Already trained. Skipping...")
            return (
                self.get_filename_without_ext(
                    epochs=self.get_current_epoch(pickle_history),
                    hyper_params=hyper_params,
                    kfold_case=kfold_case,
                ),
                pickle_history,
            )

        logger.info(f"Start training: {model_config}")
        if validation_data is None:
            if isinstance(x_train, (list, tuple)):
                random_state_val = np.random.randint(0, int(1e5))
                splits = [
                    train_test_split(
                        x, test_size=val_split, random_state=random_state_val
                    )
                    for x in x_train
                ]
                x_train = [s[0] for s in splits]
                x_val = [s[1] for s in splits]
                y_train, y_val = train_test_split(
                    y_train,
                    test_size=val_split,
                    random_state=random_state_val,
                )
            else:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train, test_size=val_split
                )
            validation_data = (x_val, y_val)
        hist = model.fit(
            x_train,
            y_train,
            epochs=model_config.epochs,
            verbose=0,  # type: ignore
            callbacks=self.create_callbacks(
                start_epoch=self.get_current_epoch(pickle_history),
                patience=model_config.patience,
                print_per_epoch=model_config.print_per_epoch,
            ),
            batch_size=model_config.batch_size,
            validation_data=validation_data,
        )
        train_output = self.get_train_output(hist.history)
        best_losses = {
            key: np.min(train_output[key], axis=0)
            for key in train_output.keys()
        }
        logger.info(f"Best losses: {json.dumps(best_losses, indent=2)}")

        pickle_history = self.update_pickle_history(
            train_output, pickle_history
        )
        train_file_stem = self.get_filename_without_ext(
            epochs=self.get_current_epoch(pickle_history),
            hyper_params=hyper_params,
            kfold_case=kfold_case,
        )
        with open("model_summary_train.txt", "w") as f:
            f.write(str(model.get_weights()))
        model.save(train_file_stem + ".keras")
        dump_pickle(Path(train_file_stem + ".pkl"), pickle_history)
        return train_file_stem, pickle_history

    def hyper_train(
        self,
        all_hyper_params: Optional[HyperParamsDictAll] = None,
    ) -> List[Tuple[str, PickleHistory]]:
        all_hyper_params = all_hyper_params or {}
        product = tuple(itertools.product(*all_hyper_params.values()))
        logger.critical(
            f"model: {self._model_name} with {len(product)} cases"  # noqa: E501
        )
        product_hyper_params = [
            dict(zip(all_hyper_params.keys(), combined_hyper_params))
            for combined_hyper_params in product
        ]
        if self.use_multiprocessing:
            logger.critical("training with multiprocessing...")
            with multiprocessing.Pool(processes=self.workers) as pool:
                results = pool.map(self.train, product_hyper_params)
        else:
            logger.critical("training without multiprocessing...")
            results = [
                self.train(hyper_params=hyper_param)
                for hyper_param in product_hyper_params
            ]

        train_file_stems = []  # type: List[str]
        pickled_histories = []  # type: List[PickleHistory]
        for result in results:
            if isinstance(result, tuple):
                train_file_stem, pickle_history = result
                pickled_histories.append(pickle_history)
                train_file_stems.append(train_file_stem)
            else:
                pickled_histories.extend([phist for _, phist in result])
                train_file_stems.extend([fstem for fstem, _ in result])

        dump_pickle(
            Path(self.get_filename_without_ext(add_datetime=True) + ".pkl"),
            pickled_histories,
        )
        assert len(train_file_stems) == len(pickled_histories), (
            f"train_file_stems: {len(train_file_stems)}, "
            f"pickled_histories: {len(pickled_histories)}"
        )
        return list(zip(train_file_stems, pickled_histories))

    def create_callbacks(
        self,
        start_epoch: int = 0,
        patience: Optional[int] = None,
        print_per_epoch: Optional[int] = None,
    ) -> List[keras.callbacks.Callback]:
        callbacks = []  # type: List[keras.callbacks.Callback]
        if patience is not None:
            callbacks.append(EarlyStopping(patience=patience))
        if print_per_epoch is not None:
            callbacks.append(
                AccuracyPerEpoch(
                    start_epoch=start_epoch,
                    print_per_epoch=print_per_epoch,
                )
            )
        return callbacks

    def create_model_and_history(
        self,
        model_config: BaseModelConfig,
        hyper_params: Optional[HyperParamsDict] = None,
        kfold_case: Optional[int] = None,
    ) -> Tuple[keras.Model, PickleHistory]:
        last_epoch, matched_stem = self.find_stem_of_last_epoch(
            self.get_filename_without_ext(
                epochs=model_config.epochs,
                hyper_params=hyper_params,
                kfold_case=kfold_case,
            )
            + ".keras"
        )
        if last_epoch is not None and matched_stem is not None:
            model = keras.models.load_model(matched_stem + ".keras")
            if model is None:
                raise ValueError(f"Model not found: {matched_stem}")
            logger.info(
                f"Loading model [{matched_stem}] with last: {last_epoch}"
            )
            return model, load_pickle(Path(matched_stem + ".pkl"))
        return self.model_class(model_config), PickleHistory(
            train_input=TrainInput(
                hyper_params=hyper_params or {}, config=model_config
            ),
            train_output=TrainOutput(),
        )

    def get_filename_without_ext(
        self,
        epochs: Optional[int] = None,
        hyper_params: Optional[Mapping[str, Union[int, float]]] = None,
        kfold_case: Optional[int] = None,
        add_datetime: bool = False,
    ) -> str:
        model_name = self._model_name
        kfold_splits = self.kfold_splits
        path = Path(self.output_path)
        path.mkdir(exist_ok=True, parents=True)
        filename = f"{model_name}"
        if epochs is not None:
            filename += f"_E{epochs}"
        if hyper_params:
            # Sort hyper_params by key
            hyper_params = dict(
                sorted((hyper_params or {}).items(), key=lambda x: x[0])
            )
            filename += "".join(
                f"[{key.upper()}={value}]"
                for key, value in hyper_params.items()
            )
        if kfold_case is not None:
            filename += f"_K{kfold_case}of{kfold_splits}"
        if add_datetime:
            filename += f"_{datetime.now():%Y_%m_%d_%H%M%S}"
        return str(path / filename)

    def apply_hyper_params(
        self, hyper_params: Optional[HyperParamsDict] = None
    ) -> BaseModelConfig:
        model_config = deepcopy(self.model_config)
        if hyper_params is not None:
            for key, value in hyper_params.items():
                assert hasattr(
                    model_config, key
                ), f"{key} is not in {model_config}"
                setattr(model_config, key, value)
        return model_config

    @staticmethod
    def get_train_output(
        hist_history: Dict[str, List[float]]
    ) -> TrainOutput:
        if "mse" in hist_history:
            hist_history["rmse"] = np.sqrt(hist_history["mse"]).tolist()
            hist_history.pop("mse")
        return TrainOutput(**hist_history)

    @staticmethod
    def update_pickle_history(
        train_output: TrainOutput, pickle_history: PickleHistory
    ) -> PickleHistory:
        for key, value in train_output.items():
            original_value = pickle_history["train_output"].get(key)
            if isinstance(original_value, list):
                pickle_history["train_output"][key].extend(value)
            elif isinstance(original_value, np.ndarray) and isinstance(
                value, np.ndarray
            ):
                pickle_history["train_output"][key] = np.concatenate(
                    (original_value, value)
                )
            else:
                if original_value is not None:
                    logger.warning(
                        f"original_value: {original_value}, value: {value}"
                    )
                pickle_history["train_output"][key] = value

        pickle_history["train_output"] = train_output
        return pickle_history

    @staticmethod
    def find_stem_of_last_epoch(
        file_path: str,
    ) -> Union[Tuple[int, str], Tuple[None, None]]:
        folder_path = Path(file_path).parent
        file_stem = Path(file_path).stem
        epoch_re = re.compile(r"_E(\d+)")

        # 에포크 정보를 제거합니다.
        epoch_removed_file_stem = epoch_re.sub("", file_stem)
        max_epoch: int = -1
        matched_stem: Optional[str] = None

        # 폴더 내의 모든 파일 검색
        for _file_path in folder_path.glob("*.pkl"):
            epoch_removed_stem = epoch_re.sub("", _file_path.stem)
            # 같은 이름 패턴이지만 에포크가 다른 파일을 찾습니다.
            if epoch_removed_stem != epoch_removed_file_stem:
                continue
            match = epoch_re.search(_file_path.stem)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch <= max_epoch:
                continue
            max_epoch = epoch
            matched_stem = _file_path.stem

        if max_epoch != -1 and matched_stem is not None:
            return max_epoch, str(folder_path / matched_stem)
        else:
            return None, None

    @staticmethod
    def get_current_epoch(pickle_history: PickleHistory) -> int:
        return len(pickle_history["train_output"].get("loss", []))
