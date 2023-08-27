import multiprocessing
from nn.ann import ANN
from nn.config import ANNConfig
from nn.train import Trainer

if __name__ == "__main__":
    ann_config = ANNConfig(
        kfold_splits=0,
        print_per_epoch=100,
        epochs=10000,
        patience=500,
        lrs=(0.001, ),
        n1s=(60, 70, ),
        n2s=(50, ),
    )
    trainer = Trainer(
        ANN,
        ann_config,
        model_name="ann",
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
    )
    trainer.hyper_train(
        hyper_params={
            "lr": ann_config.lrs,
            "n1": ann_config.n1s,
            "n2": ann_config.n2s,
        },
    )

    # lrs=(0.001, 0.005, 0.01),
    # n1s=(60, 70, 80, 90, 100, 110, 120, 130),
    # n2s=(50, 60, 70, 80, 90, 100, 110),
