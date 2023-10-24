from nn.visualize import *

if __name__ == "__main__":
    for pickle_path in Path("output/MODE0").glob("*.pkl"):
        if "[" in pickle_path.name and "]" in pickle_path.name:
            continue
        plot_graphs(pickle_path)

    # 주어진 out 데이터를 사용하여 함수를 호출
    # path = "output/PIANN_E9277[LR=0.001][N1=20][N2=10][N3=5].pkl"
    # history = load_pickle(path)["train_output"]
    # plot_metrics(history, ["loss", "mae", "mape", "rmse"])
"""
    plot_val_loss(
        Path("output_ann_re"),
        Path("image_ann_re"),
        ymin=0.0,
        ymax=0.8,
        xmin=0,
        xmax=20000,
        combined=True,
    )
"""
