from nn.ann import ANN
from nn.config import ANNConfig
from nn.train import hyper_train

if __name__ == "__main__":
    ann_config = ANNConfig(print_per_epoch=2000, epochs=200000)
    lrs: tuple[float, ...] = (0.001, 0.005, 0.01)  # Learning Rates
    n1s: tuple[int, ...] = (60, 70, 80, 90, 100, 110, 120, 130)
    n2s: tuple[int, ...] = (50, 60, 70, 80, 90, 100, 110)

    hyper_train(ANN, ann_config, lr=lrs, n1=n1s, n2=n2s)
