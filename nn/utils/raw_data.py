import os
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import pandas as pd

from nn.utils.logger import ApiLogger

logger = ApiLogger(__name__)


class SSCurve(TypedDict):
    strain: List[float]
    stress: List[float]


class SSData(TypedDict):
    idx: int
    curve: SSCurve


def read_single_ss_curve(
    csv_file_path: os.PathLike,
) -> pd.DataFrame:
    assert Path(csv_file_path).exists(), f"{csv_file_path} does not exist"
    df = pd.read_csv(csv_file_path, header=None, encoding="cp949")

    # 두 번째 행을 기반으로 열 이름 설정
    df.columns = ["Name"] + df.iloc[0].tolist()[1:]
    df = df.drop([0, 1]).reset_index(drop=True)
    df.set_index("Name", inplace=True)

    return df


def read_all_ss_curves(
    csv_directory_path: os.PathLike,
) -> Dict[int, List[SSData]]:
    ss_data_dict: Dict[int, List[SSData]] = {}
    for csv_file_path in Path(csv_directory_path).glob("*.csv"):
        seperated: List[str] = csv_file_path.stem.split("_")
        _case: int = int(seperated[0])
        _idx: int = int(seperated[1])
        assert len(seperated) == 2, f"{csv_file_path} is not valid"

        df = read_single_ss_curve(csv_file_path)
        try:
            strain = df["변형율"].tolist()
            stress = df["강도"].tolist()
            data = SSData(
                idx=_idx, curve=SSCurve(strain=strain, stress=stress)
            )
        except KeyError:
            logger.error(f"{csv_file_path} is not valid")
            continue

        if _case not in ss_data_dict:
            ss_data_dict[_case] = [data]
        else:
            ss_data_dict[_case].append(data)

    # Sort
    for _case in ss_data_dict.keys():
        ss_data_dict[_case].sort(key=lambda x: x["idx"])
    ss_data_dict = dict(sorted(ss_data_dict.items()))
    return ss_data_dict


def read_overall_table(
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
    return (
        df.iloc[:, x_indices],
        df.iloc[:, y_indices].loc[:, ["strength", "lengthavg"]],
    )
