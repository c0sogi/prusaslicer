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

from nn.config import LSTMModelConfig
from nn.dataloader import DataLoader
from nn.inference import inference
from nn.lstm import EmbeddingAttentionLSTMRegressor
from nn.schemas import (
    normalize_1d_sequence,
    read_all,
)
from nn.train import Trainer
from nn.utils.logger import ApiLogger

logger = ApiLogger(__name__)

# ========== 학습 파라미터 ========== #
EPOCHS = 1000  # 학습 횟수
SEQ_LEN = 64  # SS-curve의 길이
BATCH_SIZE = 1  # 배치 사이즈
ANN_MODEL_PATH = "ANN_E10000[LR=0.001][N1=10][N2=10][N3=10].keras"  # ANN 모델 경로
ANN_MODEL_PATH = None
OUTPUT_PATH = f".tmp/{uuid4().hex}"  # 모델 저장 경로
PATIENCE = EPOCHS // 10  # 조기 종료 기준
PRINT_PER_EPOCH = EPOCHS // 100  # 학습 횟수 당 로그 출력 횟수
ANNInputParams = [  # ANN 모델의 입력 파라미터
    "bedtemp",
    "exttemp",
    "layerthickness",
    "infillspeed",
    "density",
    "thermalresistance",
    "impactstrength",
    "glasstransitiontemp",
    "thermalconductivity",
    "linearthermalexpansioncoefficient",
]
LSTMInputParams = ["stress"] + ANNInputParams  # LSTM 모델의 입력 파라미터
LSTMOutputParams = ["stress"]  # LSTM 모델의 출력 파라미터
# ================================== #


class TestLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.model_class = EmbeddingAttentionLSTMRegressor
        self.input_params = LSTMInputParams
        self.output_params = LSTMOutputParams
        self.model_config = LSTMModelConfig(
            output_path=f".tmp/{uuid4().hex}",
            metrics=["mse", "mae"],
            kfold_splits=0,
            print_per_epoch=PRINT_PER_EPOCH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            patience=PATIENCE,
            loss_funcs=["mse"],
            loss_weights=[1.0],
            l1_reg=None,
            l2_reg=None,
            dropout_rate=0.0,
            normalize_layer=False,
            dim_out=1,
            seq_len=SEQ_LEN,
            ann_model_path=ANN_MODEL_PATH,
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
        n = 5

        def extract_points(y: np.ndarray):
            gap = seq_len // (n - 1)
            last_idx = seq_len - 1
            return tuple(y[0, min(i * gap, last_idx), 0] for i in range(n))

        y_pred_points = extract_points(y_pred)[1:]
        y_test_points = extract_points(y_test)[1:]
        print(f"prediction: {y_pred_points}, true: {y_test_points}")
        for yp, yt in zip(y_pred_points, y_test_points):
            self.assertAlmostEqual(yp, yt, delta=yt * 0.5)

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

if __name__ == "__main__":
    # test_train_and_inference 수행
    suite = unittest.TestSuite()
    suite.addTest(TestLSTM('test_train_and_inference'))
    unittest.TextTestRunner().run(suite)