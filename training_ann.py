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
    read_all_no_ss,
)
from nn.train import Trainer
from nn.utils.logger import ApiLogger

logger = ApiLogger(__name__)

# ========== 학습 파라미터 ========== #
parser = argparse.ArgumentParser(description="CLI arguments for the script")
parser.add_argument(
    "--epochs", type=int, default=10000, help="Number of epochs for training"
)
parser.add_argument(
    "--patience", type=int, default=1000, help="Number of early stopping epochs for training"
)
parser.add_argument(
    "--batch_size", type=int, default=1000, help="Batch size for training"
)
parser.add_argument(
    "--output_path",
    type=str,
    default=f"",
    help="Path to save the model",
)
parser.add_argument(
    "--use_multiprocessing",
    action="store_true",
    help="Whether to use multiprocessing",
)
parser.add_argument(
    "--lr",
    nargs="+",
    type=float,
    default=[0.001, 0.005, 0.01],
    help="Learning rate values. Example: --lr 0.001 0.005 0.01",
)
parser.add_argument(
    "--n1",
    nargs="+",
    type=int,
    default=[20, 30, 40],
    help="Values for n1. Example: --n1 20 30 40",
)
parser.add_argument(
    "--n2",
    nargs="+",
    type=int,
    default=[10, 20, 30],
    help="Values for n2. Example: --n2 10 20 30",
)
parser.add_argument(
    "--n3",
    nargs="+",
    type=int,
    default=[5, 10, 15, 20],
    help="Values for n3. Example: --n3 5 10 15 20",
)
parser.add_argument(
    "--dropout_rate",
    nargs="+",
    type=float,
    default=[0.0, 0.3, 0.5],
    help="Values for dropout_rate. Example: --dropout_rate 0.0 0.3 0.5",
)
parser.add_argument(
    "--mode",
    type=int,
    default=0,
    help="Mode for training. 0: ABSPLA vs ABSPLA, 1: ABSPLA vs PETG, 2: ABSPLA+PETG vs PETG",
)
# ================================== #

# ========== 건드리지 마세요 ========== #
args = parser.parse_args()
MODE = args.mode  # 학습 모드
EPOCHS = args.epochs  # 학습 횟수
BATCH_SIZE = args.batch_size  # 배치 사이즈
ALL_HYPER_PARAMS = {
    "lr": args.lr,
    "n1": args.n1,
    "n2": args.n2,
    "n3": args.n3,
}
OUTPUT_PATH = (
    args.output_path if args.output_path else f"./output/MODE{MODE}"
)  # 모델 저장 경로
PATIENCE = args.patience  # 조기 종료 기준
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
    # "elongation",
]


# ================================== #
def ABSPLA_vs_ABSPLA(
    model_config, input_params, output_params
):
    # 데이터셋 로드
    dataset = read_all(table_filename="table.csv", dropna=True)

    # 학습 X, Y 데이터셋 분리
    train_inputs = dataset[input_params].astype(float)
    train_inputs = train_inputs[~train_inputs.index.duplicated(keep="first")]
    train_outputs = dataset[output_params].astype(float)
    train_outputs = train_outputs[
        ~train_outputs.index.duplicated(keep="first")
    ]

    # 데이터셋 로더 생성
    data_loader = DataLoader(
        model_config=model_config,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_input_params=input_params,
        train_output_params=output_params,
    )
    print(train_inputs.shape, train_outputs.shape)
    return data_loader, None


def ABSPLA_vs_PETG(
    model_config, input_params, output_params
):
    # 데이터셋 로드
    dataset = read_all(table_filename="table.csv", dropna=True)
    petg_dataset = read_all_no_ss(table_filename="petg_table.csv")

    # 학습 X, Y 데이터셋 분리
    train_inputs = dataset[input_params].astype(float)
    train_inputs = train_inputs[~train_inputs.index.duplicated(keep="first")]
    train_outputs = dataset[output_params].astype(float)
    train_outputs = train_outputs[
        ~train_outputs.index.duplicated(keep="first")
    ]

    # 검증 데이터셋 분리
    validation_inputs = petg_dataset[input_params].astype(float)
    validation_inputs = validation_inputs[
        ~validation_inputs.index.duplicated(keep="first")
    ]
    validation_outputs = petg_dataset[output_params].astype(float)
    validation_outputs = validation_outputs[
        ~validation_outputs.index.duplicated(keep="first")
    ]

    # 데이터셋 로더 생성
    data_loader = DataLoader(
        model_config=model_config,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_input_params=input_params,
        train_output_params=output_params,
    )
    validation_data_loader = DataLoader(
        model_config=model_config,
        train_inputs=validation_inputs,
        train_outputs=validation_outputs,
        train_input_params=input_params,
        train_output_params=output_params,
    )
    print(train_inputs.shape, train_outputs.shape)
    return data_loader, validation_data_loader


def ABSPLApetg_vs_PETG(
    model_config, input_params, output_params
):
    # 데이터셋 로드
    dataset = read_all(table_filename="table.csv", dropna=True)
    petg_dataset = read_all_no_ss(table_filename="petg_table.csv")

    # 학습 X, Y 데이터셋 분리
    train_inputs = dataset[input_params].astype(float)
    train_inputs = train_inputs[~train_inputs.index.duplicated(keep="first")]
    train_outputs = dataset[output_params].astype(float)
    train_outputs = train_outputs[
        ~train_outputs.index.duplicated(keep="first")
    ]

    # 검증 데이터셋 분리
    validation_inputs = petg_dataset[input_params].astype(float)
    validation_inputs = validation_inputs[
        ~validation_inputs.index.duplicated(keep="first")
    ]
    validation_outputs = petg_dataset[output_params].astype(float)
    validation_outputs = validation_outputs[
        ~validation_outputs.index.duplicated(keep="first")
    ]

    # 데이터셋 로더 생성
    data_loader = DataLoader(
        model_config=model_config,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_input_params=input_params,
        train_output_params=output_params,
    )
    validation_data_loader = DataLoader(
        model_config=model_config,
        train_inputs=validation_inputs,
        train_outputs=validation_outputs,
        train_input_params=input_params,
        train_output_params=output_params,
    )
    print(train_inputs.shape, train_outputs.shape)
    return data_loader, validation_data_loader


class TestANN(unittest.TestCase):
    def setUp(self) -> None:
        if MODE == 0:
            dataload_callback = ABSPLA_vs_ABSPLA
        elif MODE == 1:
            dataload_callback = ABSPLA_vs_PETG
        elif MODE == 2:
            dataload_callback = ABSPLApetg_vs_PETG
        else:
            raise ValueError("Invalid MODE")
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
        logger.debug(f"all_hyper_params: {ALL_HYPER_PARAMS}")
        (
            self.data_loader,
            self.validation_data_loader,
        ) = dataload_callback(
            model_config=self.model_config,
            input_params=self.input_params,
            output_params=self.output_params,
        )
        self.trainer = Trainer(
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            model_class=self.model_class,
            model_name=self.model_class.__name__,
            model_config=self.model_config,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=USE_MULTIPROCESSING,
        )

    def test_train_and_inference(self):
        num_hyper_params = reduce(
            lambda x, y: x * len(y), ALL_HYPER_PARAMS.values(), 1
        )
        for fstem, phist in self.trainer.hyper_train(ALL_HYPER_PARAMS):
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
