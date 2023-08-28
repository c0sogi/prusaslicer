# flake8: noqa: E402
from typing import Any, Dict, List

from matplotlib import pyplot as plt
import numpy as np


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
        return [(value - min(values)) / (max(values) - min(values)) for value in values]

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


def select_top_cases(data: List[Dict[str, Any]], n: int = 1) -> List[Dict[str, Any]]:
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
        return [(value - min(values)) / (max(values) - min(values)) for value in values]

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
def plot_graphs(data_list):
    for metric in ['loss', 'mae', 'mape', 'val_loss', 'val_mse', 'val_mae', 'val_mape', 'rmse']:
        plt.figure(figsize=(12, 6))
        plt.title(f"{metric} vs Hyperparameters")
        plt.xlabel("Hyperparameters")
        plt.ylabel(metric)
        
        for data in data_list:
            hyper_params = data['train_input']['hyper_params']
            output_metrics = data['train_output']
            if metric in output_metrics:
                last_metric_value = output_metrics[metric][-1]  # Using the last value of the list
                hyper_params_str = ', '.join(f"{k}={v}" for k, v in hyper_params.items())
                plt.bar(hyper_params_str, last_metric_value, label=hyper_params_str)
        
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

plot_graphs(data_list)