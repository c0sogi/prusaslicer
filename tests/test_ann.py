import json
import multiprocessing
import unittest
from uuid import uuid4

from nn.ann import ANN
from nn.config import ANNModelConfig
from nn.dataloader import DataLoaderANN
from nn.inference import inference
from nn.train import Trainer


class TestANN(unittest.TestCase):
    def test_train(self):
        epochs = 10000
        print_per_epoch = patience = epochs // 10
        ann_model_config = ANNModelConfig(
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
        ann_all_hyper_params = {
            "lr": (0.001,),
            "n1": (20,),
            "n2": (10,),
            "n3": (5, 10),
        }
        ann_trainer = Trainer(
            data_loader_class=DataLoaderANN,
            model_class=ANN,
            model_name=ANN.__name__,
            model_config=ann_model_config,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=True,
        )
        x_test = ann_trainer.train_data.iloc[0, :].to_numpy()
        y_test = ann_trainer.train_label.iloc[0, :].to_numpy()
        for fstem, phist in ann_trainer.hyper_train(ann_all_hyper_params):
            json.dumps(phist["train_output"], indent=4)
            y_pred = inference(f"{fstem}.keras", x_test.reshape(1, -1))
            print(f"{fstem} prediction: {y_pred}, true: {y_test}")
