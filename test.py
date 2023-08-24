from nn.ann import ANN
from nn.config import ANNConfig
from nn.train import hyper_train

if __name__ == "__main__":
    ann_config = ANNConfig(
        print_per_epoch=10,
        epochs=50,
        lrs=(0.001, 0.005, 0.01),
        n1s=(60, 70, 80, 90, 100, 110, 120, 130),
        n2s=(50, 60, 70, 80, 90, 100, 110),
    )

    hyper_train(
        ANN,
        ann_config,
        lrs=ann_config.lrs,
        n1s=ann_config.n1s,
        n2s=ann_config.n2s,
    )
