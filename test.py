from nn.ann import ANN
from nn.config import ANNConfig
from nn.train import hyper_train

if __name__ == "__main__":
    ann_config = ANNConfig()

    hyper_train(
        ANN,
        ann_config,
        lr=ann_config.lrs,
        n1=ann_config.n1s,
        n2=ann_config.n2s,
    )
