from functools import reduce
import json
import multiprocessing
import unittest
from uuid import uuid4

from nn.ann import ANN
from nn.config import ANNModelConfig, LSTMModelConfig
from nn.dataloader import DataLoaderANN, DataLoaderLSTM
from nn.inference import inference
from nn.lstm import LSTM
from nn.train import Trainer


class TestANN(unittest.TestCase):
    def test_train_and_inference(self):
        epochs = 10000
        print_per_epoch = patience = epochs // 10
        model_class = ANN
        data_loader_class = DataLoaderANN

        model_config = ANNModelConfig(
            input_path="./raw_data.csv",
            output_path=f".tmp/{uuid4().hex}",
            metrics=["mse", "mae", "mape"],
            kfold_splits=0,
            print_per_epoch=print_per_epoch,
            batch_size=100,
            epochs=epochs,
            patience=patience,
            loss_funcs=["mape", "mae"],
            loss_weights=[0.5, 0.5],
            l1_reg=None,
            l2_reg=None,
            dropout_rate=0.0,
            normalize_layer=False,
            dim_out=2,
        )
        all_hyper_params = {
            "lr": (0.001,),
            "n1": (20,),
            "n2": (10,),
            "n3": (5, 10),
        }
        trainer = Trainer(
            data_loader_class=data_loader_class,
            model_class=model_class,
            model_name=model_class.__name__,
            model_config=model_config,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=True,
        )
        x_test = trainer.train_data.iloc[0, :].to_numpy()
        y_test = trainer.train_label.iloc[0, :].to_numpy()
        num_hyper_params = reduce(
            lambda x, y: x * len(y), all_hyper_params.values(), 1
        )
        for fstem, phist in trainer.hyper_train(all_hyper_params):
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
    def test_train_and_inference(self):
        epochs = 10000
        print_per_epoch = patience = epochs // 10
        model_class = LSTM
        data_loader_class = DataLoaderLSTM

        model_config = LSTMModelConfig(
            input_path="./raw_ssdata",
            output_path=f".tmp/{uuid4().hex}",
            metrics=["mse", "mae", "mape"],
            kfold_splits=0,
            print_per_epoch=print_per_epoch,
            batch_size=100,
            epochs=epochs,
            patience=patience,
            loss_funcs=["mape", "mae"],
            loss_weights=[0.5, 0.5],
            l1_reg=None,
            l2_reg=None,
            dropout_rate=0.0,
            normalize_layer=False,
            dim_out=2,
        )
        all_hyper_params = {
            "lr": (0.001,),
            "n1": (20,),
            "n2": (10,),
            "n3": (5, 10),
        }
        trainer = Trainer(
            data_loader_class=data_loader_class,
            model_class=model_class,
            model_name=model_class.__name__,
            model_config=model_config,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=True,
        )
        x_test = trainer.train_data.iloc[0, :].to_numpy()
        y_test = trainer.train_label.iloc[0, :].to_numpy()
        num_hyper_params = reduce(
            lambda x, y: x * len(y), all_hyper_params.values(), 1
        )
        for fstem, phist in trainer.hyper_train(all_hyper_params):
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
