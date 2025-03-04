import os
from pathlib import Path
from re import findall
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .typings import SSCurve
from .utils.logger import ApiLogger

logger = ApiLogger(__name__)

ANNInputParams = [
    "bedtemp",
    "exttemp",
    "layerthickness",
    "infillspeed",
    "density",
    "thermalresistance",
    "impactstrength",
    "glasstransitiontemp",
    "thermalconductivity",
    "linearthermalexpansioncoefficient",
]

ANNOutputParams = [
    # "weight",
    # "width1",
    # "width2",
    # "width3",
    # "height",
    # "depth",
    "strength",
    "lengthavg",
]


# _ANN_INPUT_PARAM_ARGS = get_args(ANNInputParams)
# ANN_INPUT_PARAM_INDICES = (
#     _ANN_INPUT_PARAM_ARGS.index("bedtemp"),
#     _ANN_INPUT_PARAM_ARGS.index("exttemp"),
#     _ANN_INPUT_PARAM_ARGS.index("layerthickness"),
#     _ANN_INPUT_PARAM_ARGS.index("infillspeed"),
#     _ANN_INPUT_PARAM_ARGS.index("density"),
#     _ANN_INPUT_PARAM_ARGS.index("thermalresistance"),
#     _ANN_INPUT_PARAM_ARGS.index("impactstrength"),
#     _ANN_INPUT_PARAM_ARGS.index("glasstransitiontemp"),
#     _ANN_INPUT_PARAM_ARGS.index("thermalconductivity"),
#     _ANN_INPUT_PARAM_ARGS.index("linearthermalexpansioncoefficient"),
# )

LSTMInputParams = ["stress"] + ANNInputParams
LSTMOutputParams = ["stress"]


def select_rows_based_on_last_index(
    df: pd.DataFrame, last_indices: List[int]
) -> pd.DataFrame:
    # 주어진 dataframe에서 인덱스 값을 추출
    indices = df.index.tolist()

    # 마지막 %d 부분이 last_indices에 포함되어 있는지 확인
    selected_indices = [
        idx for idx in indices if int(idx.split("-")[-1]) in last_indices
    ]

    # 해당하는 행만 선택하여 반환
    return df.loc[selected_indices]


def normalize_1d_sequence(
    sequence: np.ndarray, trg_len: int, normalize: bool = True
) -> np.ndarray:
    assert len(sequence.shape) in (1, 2), sequence.shape
    if len(sequence.shape) == 1:
        src_len = len(sequence)
        f = interp1d(
            np.linspace(0, src_len - 1, src_len),
            sequence,
            kind="linear",
        )
        new_y = f(np.linspace(0, src_len - 1, trg_len))
        if normalize:
            new_y = (new_y - np.min(new_y)) / (np.max(new_y) - np.min(new_y))
        return new_y.reshape(trg_len, 1)
    else:
        x = sequence[:, 0]
        y = sequence[:, 1]
        new_x = np.linspace(x.min(), x.max(), trg_len)
        new_y = interp1d(x, y)(new_x)
        if normalize:
            new_y = (new_y - new_y.min()) / (new_y.max() - new_y.min())
        return new_y.reshape(trg_len, 1)


def _read_single_ss_curve(
    csv_file_path: os.PathLike,
) -> pd.DataFrame:
    assert Path(csv_file_path).exists(), f"{csv_file_path} does not exist"
    df = pd.read_csv(csv_file_path, header=None, encoding="cp949")

    # 두 번째 행을 기반으로 열 이름 설정
    df.columns = ["Name"] + df.iloc[0].tolist()[1:]
    df = df.drop([0, 1]).reset_index(drop=True)
    return df


def _read_ss_curves(
    raw_data_path: os.PathLike,
) -> pd.DataFrame:
    ss_data_dict: Dict[str, SSCurve] = {}
    for csv_dir_path in Path(raw_data_path).iterdir():
        for csv_file_path in csv_dir_path.glob("*.csv"):
            numbers = findall(r"(\d+).(\d+)", csv_file_path.stem)
            if numbers:
                before_dot, after_dot = numbers[0]
                key = f"{csv_dir_path.name.upper()}-{before_dot}-{after_dot}"
            else:
                logger.error(f"{csv_file_path} is not valid")
                continue
            df = _read_single_ss_curve(csv_file_path)
            try:
                strain = df["변형율"].tolist()
                stress = df["강도"].tolist()
                ss_data_dict[key] = SSCurve(strain=strain, stress=stress)
            except KeyError:
                logger.error(f"{csv_file_path} is not valid")
                continue
    frames = []  # type: List[pd.DataFrame]
    for sample_name, curve in ss_data_dict.items():
        df = pd.DataFrame(curve)
        df["Name"] = sample_name
        df.set_index("Name", inplace=True)
        frames.append(df)
    return pd.concat(frames)


def _read_x_and_y_from_table(
    csv_file_path: os.PathLike,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_file_path, header=None)

    x_indices: pd.Index = df.columns[df.iloc[0] == "X"] - 1
    y_indices: pd.Index = df.columns[df.iloc[0] == "Y"] - 1

    # 두 번째 행을 기반으로 열 이름 설정
    df.columns = ["Name"] + df.iloc[1].tolist()[1:]
    df = df.drop([0, 1]).reset_index(drop=True)

    # 'Name' 열의 값을 인덱스로 설정
    df.set_index("Name", inplace=True)
    x = df.iloc[:, x_indices]
    y = df.iloc[:, y_indices]
    assert x.shape[0] == y.shape[0], f"{x.shape} != {y.shape}"
    return x, y


def group_ss_curves(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            df.groupby(df.index)["strain"].apply(np.array),
            df.groupby(df.index)["stress"].apply(np.array),
            df.drop(columns=["strain", "stress"]).groupby(df.index).first(),
        ],
        axis=1,
    )


def read_all(
    raw_data_dir: os.PathLike = Path("./raw_data"),
    table_filename: str = "table.csv",
    dropna: bool = True,
) -> pd.DataFrame:
    # Load raw data (ss curves and table)
    ss_curves = _read_ss_curves(raw_data_dir)
    x_data, y_data = _read_x_and_y_from_table(
        Path(raw_data_dir) / table_filename
    )
    for to_merge in (x_data, y_data):
        ss_curves = ss_curves.merge(to_merge, on="Name", how="left")
    if dropna:
        ss_curves.dropna(inplace=True)

    logger.debug(f"===== Number of valid data: {ss_curves.shape[0]} =====")
    return group_ss_curves(ss_curves)


def read_all_no_ss(
    raw_data_dir: os.PathLike = Path("./raw_data"),
    table_filename: str = "petg_table.csv",
) -> pd.DataFrame:
    # Load raw data (ss curves and table)
    x_data, y_data = _read_x_and_y_from_table(
        Path(raw_data_dir) / table_filename
    )
    all_data = pd.concat([x_data, y_data], axis=1)
    logger.debug(f"===== Number of valid data: {all_data.shape[0]} =====")
    return all_data


# def read_sscurves(
#     raw_data_dir: os.PathLike = Path("./raw_data"),
#     dropna: bool = True,
# ) -> pd.DataFrame:
#     # Load raw data (ss curves and table)
#     ss_curves = _read_ss_curves(raw_data_dir)
#     x_data, y_data = _read_x_and_y_from_table(
#         Path(raw_data_dir) / table_filename
#     )
#     # for to_merge in (x_data, y_data):
#     #     ss_curves = ss_curves.merge(to_merge, on="Name", how="left")
#     if dropna:
#         ss_curves.dropna(inplace=True)

#     logger.debug(f"===== Number of valid data: {ss_curves.shape[0]} =====")
#     return group_ss_curves(ss_curves)
