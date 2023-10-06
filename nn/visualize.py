# flake8: noqa: E402
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt

from .dataloader import load_pickle_list, load_pickle
from .train import PickleHistory


def visualize(
    data: List[Dict[str, Any]],
) -> None:
    cases = [item["case"] for item in data]
    mae_values = [item["mae"] for item in data]
    mape_values = [item["mape"] for item in data]
    rmse_values = [item["rmse"] for item in data]
    # Plotting the data
    # Plotting the data with separate axes for each metric

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # MAE plot
    axs[0].plot(cases, mae_values, marker="o", color="blue", label="MAE")
    axs[0].set_ylabel("MAE Value")
    axs[0].set_title("MAE vs Case Number")
    axs[0].grid(True)
    axs[0].legend()

    # MAPE plot
    axs[1].plot(cases, mape_values, marker="o", color="green", label="MAPE")
    axs[1].set_ylabel("MAPE Value (%)")
    axs[1].set_title("MAPE vs Case Number")
    axs[1].grid(True)
    axs[1].legend()

    # RMSE plot
    axs[2].plot(cases, rmse_values, marker="o", color="red", label="RMSE")
    axs[2].set_xlabel("Case Number")
    axs[2].set_ylabel("RMSE Value")
    axs[2].set_title("RMSE vs Case Number")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def visualize_normalized(
    data: List[Dict[str, Any]],
) -> None:
    cases = [item["case"] for item in data]
    mae_values = [item["mae"] for item in data]
    mape_values = [item["mape"] for item in data]
    rmse_values = [item["rmse"] for item in data]

    # Normalize function
    def normalize(values):
        return [
            (value - min(values)) / (max(values) - min(values))
            for value in values
        ]

    # Normalize the metrics
    normalized_mae = normalize(mae_values)
    normalized_mape = normalize(
        [v / 100 for v in mape_values]
    )  # Dividing by 100 to bring MAPE to the range [0,1]
    normalized_rmse = normalize(rmse_values)

    # Calculate the average of the normalized values
    average_normalized_values = np.mean(
        [normalized_mae, normalized_mape, normalized_rmse], axis=0
    )

    # Plotting the average normalized values
    plt.figure(figsize=(10, 6))
    plt.plot(
        cases,
        average_normalized_values,
        marker="o",
        color="purple",
        label="Average Normalized Value",
    )
    plt.xlabel("Case Number")
    plt.ylabel("Average Normalized Value")
    plt.title("Average Normalized Metrics vs Case Number")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


def select_top_cases(
    data: List[Dict[str, Any]], n: int = 1
) -> List[Dict[str, Any]]:
    """
    Selects the top n cases with the lowest average normalized metric values.

    Args:
    - data (list): List of dictionaries containing case details and metric values.
    - n (int): Number of top cases to select.

    Returns:
    - List of top n cases with the lowest average normalized metric values.
    """

    # Normalize function
    def normalize(values):
        return [
            (value - min(values)) / (max(values) - min(values))
            for value in values
        ]

    # Extracting and normalizing the metric values
    mae_values = [item["mae"] for item in data]
    mape_values = [item["mape"] for item in data]
    rmse_values = [item["rmse"] for item in data]

    normalized_mae = normalize(mae_values)
    normalized_mape = normalize(
        [v / 100 for v in mape_values]
    )  # Dividing by 100 to bring MAPE to the range [0,1]
    normalized_rmse = normalize(rmse_values)

    # Calculate the average of the normalized values
    average_normalized_values = np.mean(
        [normalized_mae, normalized_mape, normalized_rmse], axis=0
    )

    # Associate the average normalized value with the case details
    case_with_avg_normalized = [
        {"case": item["case"], "avg_normalized_value": avg_val}
        for item, avg_val in zip(data, average_normalized_values)
    ]

    # Sort the cases based on the average normalized value
    sorted_cases = sorted(
        case_with_avg_normalized, key=lambda x: x["avg_normalized_value"]
    )

    # Return the top n cases
    return sorted_cases[:n]


# Function to plot graphs
def plot_graphs(
    pickle_path: os.PathLike,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    file_stem = Path(pickle_path).stem
    pickle_data: List[PickleHistory] = load_pickle_list(pickle_path)

    for metric in pickle_data[0]["train_output"].keys():
        plt.figure(figsize=(12, 6))
        plt.title(f"{metric} vs Hyperparameters")
        plt.xlabel("Hyperparameters")
        plt.ylabel(metric)

        for data in pickle_data:
            hyper_params = data["train_input"]["hyper_params"]
            output_metrics = data["train_output"]
            if metric in output_metrics:
                last_metric_value = output_metrics[metric][
                    -1
                ]  # Using the last value of the list
                hyper_params_str = ", ".join(
                    f"{k}={v}" for k, v in hyper_params.items()
                )
                plt.bar(
                    hyper_params_str,
                    last_metric_value,
                    label=hyper_params_str,
                )

        # Save the figure

        img_filename = Path(pickle_path).parent / f"{file_stem}_{metric}.png"
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        plt.savefig(img_filename)
        plt.close()


def plot_metrics(history, metrics):
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 4, i)
        plt.plot(history[metric], label="Training {}".format(metric))

        # MAPE에 대한 y축 범위 조절
        if metric == "mape" or metric == "val_mape":
            plt.ylim(0, 150)  # 예를 들어 0%에서 150% 사이로 설정
        else:
            plt.ylim(0, 3)

        val_metric = "val_" + metric
        if val_metric in history:
            plt.plot(
                history[val_metric],
                label="Validation {}".format(metric),
                linestyle="--",
            )
        plt.legend()
        plt.title(metric)
    plt.tight_layout()
    plt.show()


def plot_val_loss(
    pkl_dir: os.PathLike,
    save_dir: os.PathLike,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    xmin: Optional[int] = None,
    xmax: Optional[int] = None,
    combined: bool = False,
):
    pkl_dir, save_dir = Path(pkl_dir), Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    num_files = len(list(pkl_dir.glob("*.pkl")))

    if combined:
        plt.figure(figsize=(12, 10))
    else:
        plt.figure(figsize=(10, 6))

    for pkl_path in pkl_dir.glob("*.pkl"):
        # 데이터 로드
        data = load_pickle(pkl_path)
        if "train_output" not in data:
            print(f"Invalid pickle file: {pkl_path}")
            continue

        assert "val_loss" in data["train_output"], "No val_loss data"
        loss = data["train_output"]["val_loss"]

        # 그래프 그리기
        plt.plot(
            loss,
            label=f"Validation Loss for {pkl_path.name}"
            if combined
            else "Training Validation Loss",
            linewidth=0.4,
        )

        if not combined:
            plt.title(f"Epoch vs Validation Loss for {pkl_path.name}")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            if ymin is not None and ymax is not None:
                plt.ylim(ymin, ymax)
            if xmin is not None and xmax is not None:
                plt.xlim(xmin, xmax)
            plt.legend(
                loc="upper left", bbox_to_anchor=(1, 1), prop={"size": 10}
            )
            # 이미지로 저장
            img_name = pkl_path.stem + ".png"
            save_path = save_dir / img_name
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    if combined:
        plt.title("Combined Epoch vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        if xmin is not None and xmax is not None:
            plt.xlim(xmin, xmax)

        # ncol 값을 항목 개수에 따라 조절
        ncol = (num_files - 1) // 60 + 1

        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            prop={"size": 8},
            ncol=ncol,
        )

        # 이미지로 저장
        save_path = save_dir / "combined_val_loss.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # for pickle_path in Path("output").glob("*.pkl"):
    #     if "[" in pickle_path.name and "]" in pickle_path.name:
    #         continue
    #     plot_graphs(pickle_path)

    # 주어진 out 데이터를 사용하여 함수를 호출
    # path = "output/PIANN_E9277[LR=0.001][N1=20][N2=10][N3=5].pkl"
    # history = load_pickle(path)["train_output"]
    # plot_metrics(history, ["loss", "mae", "mape", "rmse"])

    plot_val_loss(
        Path("output_ann_re"),
        Path("image_ann_re"),
        ymin=0.0,
        ymax=0.8,
        xmin=0,
        xmax=20000,
        combined=True,
    )
