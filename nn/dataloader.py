from typing import Iterator, Tuple

import pandas as pd
from sklearn.model_selection import KFold


def dataset_batch_iterator(
    x_data: pd.DataFrame, y_data: pd.DataFrame, batch_size: int = 1
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    dataset_size = min(len(x_data), len(y_data))
    for batch_start in range(0, dataset_size, batch_size):
        batch_end = min(dataset_size, batch_start + batch_size)
        yield x_data[batch_start:batch_end], y_data[batch_start:batch_end]


def dataset_kfold_iterator(
    x_data: pd.DataFrame, y_data: pd.DataFrame, n_splits: int = 5
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(x_data, y_data):
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        yield x_train, y_train, x_test, y_test
