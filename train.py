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
        dim_in=4,
        dim_out=1,
        kfold_splits=0,
        print_per_epoch=100,
        batch_size=100,
        epochs=10000,
        patience=500,
    )
    trainer = Trainer(
        ANN,
        model_config,
        model_name="ANN",
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
    )
    trainer.hyper_train(
        {
            "lr": (0.001, 0.005, 0.01),
            "n0": (50,),
            "n1": (60, 70, 80, 90, 100, 110, 120, 130),
            "n2": (50, 60, 70, 80, 90, 100, 110),
        },
    )

    # 로스 그래프 그리기
    for pickle_path in Path("output").glob("*.pickle"):
        if "[" in pickle_path.name and "]" in pickle_path.name:
            continue
        plot_graphs(pickle_path)
    # trainer.train(hyper_params={"lr": 0.001, "n1": 60, "n2": 50})

    # "lr": (0.001, 0.005, 0.01),
    # "n1": (60, 70, 80, 90, 100, 110, 120, 130),
    # "n2": (50, 60, 70, 80, 90, 100, 110),
