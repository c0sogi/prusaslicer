# flake8: noqa
import argparse
from functools import reduce
import json
import multiprocessing
import random
from typing import Tuple
import unittest
from uuid import uuid4
import numpy as np

import pandas as pd

from nn.ann import ANN
from nn.config import ANNModelConfig
from nn.dataloader import DataLoader
from nn.inference import inference
from nn.schemas import (
    ANNInputParams,
    ANNOutputParams,
    read_all,
)
from nn.train import Trainer
from nn.utils.logger import ApiLogger

logger = ApiLogger(__name__)
parser = argparse.ArgumentParser(description="CLI arguments for the script")
parser.add_argument(
    "--epochs", type=int, default=10000, help="Number of epochs for training"
)
parser.add_argument(
    "--batch_size", type=int, default=1000, help="Batch size for training"
)
parser.add_argument(
    "--output_path",
    type=str,
    default=f".tmp/{uuid4().hex}",
    help="Path to save the model",
)
parser.add_argument(
    "--use_multiprocessing",
    type=bool,
    default=False,
    help="Whether to use multiprocessing",
)

args = parser.parse_args()

# ========== 학습 파라미터 ========== #
EPOCHS = args.epochs  # 학습 횟수
BATCH_SIZE = args.batch_size  # 배치 사이즈
OUTPUT_PATH = args.output_path  # 모델 저장 경로
PATIENCE = EPOCHS // 10  # 조기 종료 기준
PRINT_PER_EPOCH = EPOCHS // 100  # 학습 횟수 당 로그 출력 횟수
USE_MULTIPROCESSING = (
    args.use_multiprocessing
)  # 멀티프로세싱 사용 여부 (True 사용시 CPU 사용률 100%)
ANNInputParams = [  # ANN 모델의 입력 파라미터
    "bedtemp",
    "exttemp",
    "layerthickness",
    "density",
    "thermalresistance",
    "impactstrength",
    "glasstransitiontemp",
    "thermalconductivity",
    "linearthermalexpansioncoefficient",
]
ANNOutputParams = [
    "strength",
    "lengthavg",
    "weight",
    "elongation",
]
GRID_SEARCH_HYPER_PARAMS = {
    "lr": (0.001, 0.005, 0.01),
    "n1": (20, 30, 40),
    "n2": (10, 20, 30),
    "n3": (5, 10, 15, 20),
}
# ================================== #


class TestANN(unittest.TestCase):
    def setUp(self) -> None:
        self.model_class = ANN
        self.input_params = ANNInputParams
        self.output_params = ANNOutputParams
        self.model_config = ANNModelConfig(
            output_path=OUTPUT_PATH,
            metrics=["mse", "mae", "mape"],
            kfold_splits=0,
            print_per_epoch=PRINT_PER_EPOCH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            patience=PATIENCE,
            loss_funcs=[
                "mape" if output_param == "lengthavg" else "mae"
                for output_param in ANNOutputParams
            ],
            loss_weights=[0.5 for _ in range(len(ANNOutputParams))],
            l1_reg=None,
            l2_reg=None,
            dropout_rate=0.0,
            normalize_layer=False,
            dim_out=len(ANNOutputParams),
        )
        self.all_hyper_params = GRID_SEARCH_HYPER_PARAMS
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
            use_multiprocessing=USE_MULTIPROCESSING,
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


if __name__ == "__main__":
    # test_train_and_inference 수행
    suite = unittest.TestSuite()
    suite.addTest(TestANN("test_train_and_inference"))
    unittest.TextTestRunner().run(suite)
