import multiprocessing
from nn.ann import ANN
from nn.config import ModelConfig
from nn.train import Trainer


if __name__ == "__main__":
    model_config = ModelConfig(
        input_path="./raw_data.csv",
        output_path="./output",
        metrics=["mse", "mae", "mape"],
        dim_in=50,
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
        multiple_hyper_params={
            "lr": (0.001, ),
            "n1": (60, 70,),
            "n2": (50, ),
        },
    )
    # trainer.train(hyper_params={"lr": 0.001, "n1": 60, "n2": 50})

    # "lr": (0.001, 0.005, 0.01),
    # "n1": (60, 70, 80, 90, 100, 110, 120, 130),
    # "n2": (50, 60, 70, 80, 90, 100, 110),
