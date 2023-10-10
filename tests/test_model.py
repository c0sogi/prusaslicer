# flake8: noqa
from functools import reduce
import json
import multiprocessing
import random
from typing import Iterable, List, Tuple
import unittest
from uuid import uuid4
import numpy as np

import pandas as pd

from nn.ann import ANN
from nn.config import ANNModelConfig, LSTMModelConfig
from nn.dataloader import DataLoader
from nn.inference import inference
from nn.lstm import EmbeddingAttentionLSTMRegressor
from nn.schemas import (
    ANNInputParams,
    ANNOutputParams,
    LSTMInputParams,
    LSTMOutputParams,
    normalize_1d_sequence,
    read_all,
)
from nn.train import Trainer
from nn.utils.logger import ApiLogger

logger = ApiLogger(__name__)


class TestANN(unittest.TestCase):
    def setUp(self) -> None:
        self.epoch = 10000
        self.print_per_epoch = self.epoch // 100
        self.patience = self.epoch // 10
        self.model_class = ANN
        self.input_params = ANNInputParams
        self.output_params = ANNOutputParams
        self.model_config = ANNModelConfig(
            output_path=f".tmp/{uuid4().hex}",
            metrics=["mse", "mae", "mape"],
            kfold_splits=0,
            print_per_epoch=self.print_per_epoch,
            batch_size=100,
            epochs=self.epoch,
            patience=self.patience,
            loss_funcs=["mape", "mae"],
            loss_weights=[0.5, 0.5],
            l1_reg=None,
            l2_reg=None,
            dropout_rate=0.0,
            normalize_layer=False,
            dim_out=2,
        )
        self.all_hyper_params = {
            "lr": (0.001,),
            "n1": (10,),
            "n2": (10,),
            "n3": (10,),
        }
        dataset = read_all(dropna=True)
        train_inputs = dataset[self.input_params].astype(float)
        train_outputs = dataset[self.output_params].astype(float)
        self.train_inputs = train_inputs[
            ~train_inputs.index.duplicated(keep="first")
        ]
        self.train_outputs = train_outputs[
            ~train_outputs.index.duplicated(keep="first")
        ]
        self.data_loader = DataLoader(
            model_config=self.model_config,
            train_inputs=self.train_inputs,
            train_outputs=self.train_outputs,
            train_input_params=self.input_params,
            train_output_params=self.output_params,
        )
        self.trainer = Trainer(
            data_loader=self.data_loader,
            model_class=self.model_class,
            model_name=self.model_class.__name__,
            model_config=self.model_config,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=True,
        )
        print(self.train_inputs)
        print(self.train_outputs)
        print(self.train_inputs.shape, self.train_outputs.shape)

    def test_train_and_inference(self):
        num_hyper_params = reduce(
            lambda x, y: x * len(y), self.all_hyper_params.values(), 1
        )
        for fstem, phist in self.trainer.hyper_train(self.all_hyper_params):
            num_hyper_params -= 1
            json.dumps(phist["train_output"], indent=4)
            self.test_inference(fstem + ".keras")

        self.assertEqual(num_hyper_params, 0)

    def test_inference(
        self,
        model_path: str = r".tmp\434ce9d21fab4746b794283774c0c54e\ANN_E4453[LR=0.001][N1=20][N2=10][N3=10].keras",
    ):
        x_test, y_test = self.test_data
        y_pred = inference(model_path, x_test)
        print(f"prediction: {y_pred}, true: {y_test}")
        strength_pred = float(y_pred[0][0])
        strength_true = float(y_test[0][0])
        self.assertAlmostEqual(
            strength_pred, strength_true, delta=strength_true * 0.5
        )
        dimension_pred = float(y_pred[0][1])
        dimension_true = float(y_test[0][1])
        self.assertAlmostEqual(
            dimension_pred, dimension_true, delta=dimension_true * 1.0
        )

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x_test, y_test = random.sample(
            [
                data
                for data in self.data_loader.dataset_batch_iterator(
                    batch_size=1
                )
            ],
            k=1,
        )[0]
        assert isinstance(x_test, pd.DataFrame) and isinstance(
            y_test, pd.DataFrame
        )
        return x_test.to_numpy(), y_test.to_numpy()


class TestLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.epoch = 1000
        self.print_per_epoch = 1
        self.patience = self.epoch // 10
        self.model_class = EmbeddingAttentionLSTMRegressor
        self.input_params = LSTMInputParams
        self.output_params = LSTMOutputParams
        self.model_config = LSTMModelConfig(
            output_path=f".tmp/{uuid4().hex}",
            metrics=["mse", "mae"],
            kfold_splits=0,
            print_per_epoch=self.print_per_epoch,
            batch_size=1,
            epochs=self.epoch,
            patience=self.patience,
            loss_funcs=["mse"],
            loss_weights=[1.0],
            l1_reg=None,
            l2_reg=None,
            dropout_rate=0.0,
            normalize_layer=False,
            dim_out=1,
            seq_len=64,
            ann_model_path="ANN_E10000[LR=0.001][N1=10][N2=10][N3=10].keras",
        )
        dataset = read_all(dropna=True)[self.input_params]
        encoder_inputs = (
            dataset.drop(columns=["stress"]).astype(float).to_numpy()
        )
        decoder_outputs = (
            dataset["stress"]
            .apply(
                lambda x: pd.Series(
                    normalize_1d_sequence(x, self.model_config.seq_len)
                )
            )
            .to_numpy()
        )[:, :, np.newaxis]
        decoder_inputs = np.zeros_like(decoder_outputs)
        decoder_inputs[:, 1:, :] = decoder_outputs[:, :-1, :]
        # self.train_inputs = [encoder_inputs, decoder_inputs]
        self.train_inputs = encoder_inputs
        self.train_outputs = decoder_outputs
        self.data_loader = DataLoader(
            model_config=self.model_config,
            train_inputs=self.train_inputs,
            train_outputs=self.train_outputs,
            train_input_params=self.input_params,
            train_output_params=self.output_params,
        )
        self.trainer = Trainer(
            data_loader=self.data_loader,
            model_class=self.model_class,
            model_name=self.model_class.__name__,
            model_config=self.model_config,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=False,
        )
        print(self.train_inputs)
        print(self.train_outputs)
        print(
            self.train_inputs.shape,
            self.train_outputs.shape,
        )

    def test_train_and_inference(self):
        all_hyper_params = {"seq_len": (self.model_config.seq_len,)}
        num_hyper_params = reduce(
            lambda x, y: x * len(y), all_hyper_params.values(), 1
        )
        for fstem, phist in self.trainer.hyper_train(all_hyper_params):
            num_hyper_params -= 1
            json.dumps(phist["train_output"], indent=4)
            self.test_inference(fstem + ".keras")
        #     print(f"{fstem} prediction: {y_pred}, true: {y_test}")
        #     strength_pred = float(y_pred[0][0])
        #     strength_true = float(y_test[0][0])
        #     self.assertAlmostEqual(
        #         strength_pred, strength_true, delta=strength_true * 0.5
        #     )
        #     dimension_pred = float(y_pred[0][1])
        #     dimension_true = float(y_test[0][1])
        #     self.assertAlmostEqual(
        #         dimension_pred, dimension_true, delta=dimension_true * 1.0
        #     )
        self.assertEqual(num_hyper_params, 0)

    def test_inference(
        self,
        model_path: str = r".tmp\172689c286e24684a2d2ba234ce454e6\LSTM_E556[SEQ_LEN=512].keras",
    ):
        x_test, y_test = self.test_data
        y_pred = inference(model_path, x_test)
        assert (
            y_pred.shape == y_test.shape
        ), f"{y_pred.shape} != {y_test.shape}"
        seq_len = y_pred.shape[1]
        y_pred_low, y_pred_mid, y_pred_high = (
            y_pred[0, seq_len // 4, 0],  # type: ignore
            y_pred[0, seq_len // 2, 0],  # type: ignore
            y_pred[0, seq_len * 3 // 4, 0],  # type: ignore
        )
        y_test_low, y_test_mid, y_test_high = (
            y_test[0, seq_len // 4, 0],  # type: ignore
            y_test[0, seq_len // 2, 0],  # type: ignore
            y_test[0, seq_len * 3 // 4, 0],  # type: ignore
        )
        print(f"y_pred: {y_pred_low}, {y_pred_mid}, {y_pred_high}")
        print(f"y_test: {y_test_low}, {y_test_mid}, {y_test_high}")
        self.assertAlmostEqual(
            y_pred_low, y_test_low, delta=y_test_low * 0.5
        )
        self.assertAlmostEqual(
            y_pred_mid, y_test_mid, delta=y_test_mid * 0.5
        )
        self.assertAlmostEqual(
            y_pred_high, y_test_high, delta=y_test_high * 0.5
        )

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x_test = self.train_inputs
        y_test = self.train_outputs
        assert isinstance(x_test, np.ndarray) and isinstance(
            y_test, np.ndarray
        ), f"{type(x_test)} & {type(y_test)}"
        assert (
            x_test.shape[0] == y_test.shape[0]
        ), f"{x_test.shape} != {y_test.shape}"
        random_idx = random.randint(0, x_test.shape[0] - 1)
        return (
            x_test[random_idx : random_idx + 1],
            y_test[random_idx : random_idx + 1],
        )
