# flake8: noqa
import multiprocessing
from datetime import datetime
from pathlib import Path

from nn.ann import ANN
from nn.cnn import CNN
from nn.config import ANNModelConfig, CNNModelConfig
from nn.dataloader import DataLoaderANN, DataLoaderCNN
from nn.train import Trainer
from nn.visualize import plot_graphs

if __name__ == "__main__":
    # 모델 및 트레이너에 대한 간단한 설정
    ann_model_config = ANNModelConfig(
        input_path="./raw_data.csv",
        output_path=f"./{datetime.now().strftime('%Y%m%d%H%M')}",
        metrics=["mse", "mae", "mape"],
        kfold_splits=0,
        print_per_epoch=100,
        batch_size=100,
        epochs=20000,
        patience=1000,
        loss_funcs=["mae", "mae"],
        loss_weights=[0.5, 0.5],
        l1_reg=None,
        l2_reg=None,
        dropout_rate=0.0,
        normalize_layer=False,
        dim_out=2,
    )
    cnn_model_config = CNNModelConfig(
        input_path="./raw_data.csv",
        output_path=f"./{datetime.now().strftime('%Y%m%d%H%M')}",
        metrics=["mse", "mae", "mape"],
        kfold_splits=0,
        print_per_epoch=100,
        batch_size=100,
        epochs=20000,
        patience=1000,
        loss_funcs=["mae", "mae"],
        loss_weights=[0.5, 0.5],
        l1_reg=None,
        l2_reg=None,
        dropout_rate=0.0,
        normalize_layer=False,
        dim_out=2,
    )
    ann_all_hyper_params = {
        "lr": (0.001, 0.005, 0.01),
        "n1": (20, 30, 40),
        "n2": (10, 20, 30),
        "n3": (5, 10, 15, 20),
    }
    cnn_all_hyper_params = {
        "lr": (0.001, 0.005, 0.01),
        "n1": (20, 30, 40),
        "n2": (10, 20, 30),
        "n3": (5, 10, 15, 20),
    }

    # 실제 학습
    ann_trainer = Trainer(
        data_loader_class=DataLoaderANN,
        model_class=ANN,
        model_name=ANN.__name__,
        model_config=ann_model_config,
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
    )
    cnn_trainer = Trainer(
        data_loader_class=DataLoaderCNN,
        model_class=CNN,
        model_name=CNN.__name__,
        model_config=cnn_model_config,
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
    )
    ann_trainer.hyper_train(ann_all_hyper_params)
    cnn_trainer.hyper_train(cnn_all_hyper_params)
    # for pickle_path in Path("output").glob("*.pkl"):
    #     if "[" in pickle_path.name and "]" in pickle_path.name:
    #         continue
    #     plot_graphs(pickle_path)
    # "lr": (0.001, 0.005, 0.01),
    # "n1": (60, 70, 80, 90, 100, 110, 120, 130),
    # "n2": (50, 60, 70, 80, 90, 100, 110),
