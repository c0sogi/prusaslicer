import multiprocessing
from pathlib import Path

from nn.ann import ANN
from nn.config import ModelConfig
from nn.train import Trainer
from nn.visualize import plot_graphs


if __name__ == "__main__":
    # 모델 및 트레이너에 대한 간단한 설정
    model_config = ModelConfig(
        input_path="./raw_data.csv",
        output_path="./output",
        metrics=["mse", "mae", "mape"],
        dim_out=2,
        kfold_splits=0,
        print_per_epoch=100,
        batch_size=500,
        epochs=20000,
        patience=1000,
        loss_funcs=["mae", "mae"],
        loss_weights=[0.5, 0.5],
        l1_reg=0.01,
        l2_reg=0.01,
        dropout_rate=0.5,
        normalize_layer=True,
    )
    trainer = Trainer(
        ANN,
        model_config,
        model_name="ANN_RE",
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=False,
    )
    trainer.hyper_train(
        {
            "lr": (0.001, 0.005, 0.01),
            "n1": (20, 30, 40),
            "n2": (10, 20, 30),
            "n3": (5, 10, 15, 20),
        },
    )

    # 로스 그래프 그리기
    for pickle_path in Path("output").glob("*.pkl"):
        if "[" in pickle_path.name and "]" in pickle_path.name:
            continue
        plot_graphs(pickle_path)
    # trainer.train(hyper_params={"lr": 0.001, "n1": 60, "n2": 50})

    # "lr": (0.001, 0.005, 0.01),
    # "n1": (60, 70, 80, 90, 100, 110, 120, 130),
    # "n2": (50, 60, 70, 80, 90, 100, 110),
