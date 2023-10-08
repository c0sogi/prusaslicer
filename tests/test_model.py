# flake8: noqa
from functools import reduce
import json
import multiprocessing
import unittest
from uuid import uuid4

from nn.ann import ANN
from nn.config import ANNModelConfig, LSTMModelConfig
from nn.dataloader import DataLoader, read_all
from nn.inference import inference
from nn.lstm import LSTM
from nn.schemas import (
    ANNInputParams,
    ANNOutputParams,
    LSTMInputParams,
    LSTMOutputParams,
)
from nn.train import Trainer
from nn.utils.logger import ApiLogger

logger = ApiLogger(__name__)


class TestANN(unittest.TestCase):
    def setUp(self) -> None:
        self.epoch = 10000
        self.print_per_epoch = self.patience = self.epoch // 10
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
        all_hyper_params = {
            "lr": (0.001,),
            "n1": (20,),
            "n2": (10,),
            "n3": (5, 10),
        }

        x_test = self.trainer.train_inputs.iloc[0, :].to_numpy()
        y_test = self.trainer.train_outputs.iloc[0, :].to_numpy()
        num_hyper_params = reduce(
            lambda x, y: x * len(y), all_hyper_params.values(), 1
        )
        for fstem, phist in self.trainer.hyper_train(all_hyper_params):
            num_hyper_params -= 1
            json.dumps(phist["train_output"], indent=4)
            y_pred = inference(f"{fstem}.keras", x_test.reshape(1, -1))
            print(f"{fstem} prediction: {y_pred}, true: {y_test}")
            strength_pred = y_pred[0][0]
            strength_true = y_test[0]
            self.assertAlmostEqual(
                strength_pred, strength_true, delta=strength_true * 0.5
            )
            dimension_pred = y_pred[0][1]
            dimension_true = y_test[1]
            self.assertAlmostEqual(
                dimension_pred, dimension_true, delta=dimension_true * 1.0
            )

        self.assertEqual(num_hyper_params, 0)


class TestLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.epoch = 10000
        self.print_per_epoch = self.patience = self.epoch // 10
        self.model_class = LSTM
        self.input_params = LSTMInputParams
        self.output_params = LSTMOutputParams
        self.model_config = LSTMModelConfig(
            output_path=f".tmp/{uuid4().hex}",
            metrics=["mse", "mae", "mape"],
            kfold_splits=0,
            print_per_epoch=self.print_per_epoch,
            batch_size=100,
            epochs=self.epoch,
            patience=self.patience,
            loss_funcs=["mae"],
            loss_weights=[0.5],
            l1_reg=None,
            l2_reg=None,
            dropout_rate=0.0,
            normalize_layer=False,
            dim_out=1,
        )
        dataset = read_all(dropna=True)
        self.train_inputs = dataset[self.input_params].astype(float)
        self.train_outputs = dataset[self.output_params].astype(float)
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
        all_hyper_params = {
            "lr": (0.001,),
            "n1": (20,),
            "n2": (10,),
            "n3": (5, 10),
        }

        # x_test = self.trainer.train_inputs.iloc[0, :].to_numpy()
        # y_test = self.trainer.train_outputs.iloc[0, :].to_numpy()
        # num_hyper_params = reduce(
        #     lambda x, y: x * len(y), all_hyper_params.values(), 1
        # )
        # for fstem, phist in self.trainer.hyper_train(all_hyper_params):
        #     num_hyper_params -= 1
        #     json.dumps(phist["train_output"], indent=4)
        #     y_pred = inference(f"{fstem}.keras", x_test.reshape(1, -1))
        #     print(f"{fstem} prediction: {y_pred}, true: {y_test}")
        #     strength_pred = y_pred[0][0]
        #     strength_true = y_test[0]
        #     self.assertAlmostEqual(
        #         strength_pred, strength_true, delta=strength_true * 0.5
        #     )
        #     dimension_pred = y_pred[0][1]
        #     dimension_true = y_test[1]
        #     self.assertAlmostEqual(
        #         dimension_pred, dimension_true, delta=dimension_true * 1.0
        #     )

        # self.assertEqual(num_hyper_params, 0)
