import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
)

from matplotlib import pyplot as plt
import numpy as np

T = TypeVar("T", bound=Union[str, float, timedelta])
KEY_PATTERN = re.compile(r"[^_]+_(?P<key>.*?)\.gcode")
VALUE_PATTERN = re.compile(r"^;\s*(?P<key>\S+.*?)\s*=\s*(?P<value>.+)")
DAY_PATTERN = re.compile(r"(\d+)d")
HOUR_PATTERN = re.compile(r"(\d+)h")
MINUTE_PATTERN = re.compile(r"(\d+)m")
SECOND_PATTERN = re.compile(r"(\d+)s")

# 필요한 정보만 dict에 저장합니다.
KEYS_OF_INTEREST = (
    "filament used [mm]",
    "filament used [cm3]",
    "filament used [g]",
    "estimated printing time (normal mode)",
    "estimated first layer printing time (normal mode)",
)


def validate_string_type(value: str) -> Union[str, int, float]:
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_duration(value: str) -> Optional[timedelta]:
    days_match = DAY_PATTERN.search(value)
    hours_match = HOUR_PATTERN.search(value)
    minutes_match = MINUTE_PATTERN.search(value)
    seconds_match = SECOND_PATTERN.search(value)
    if not any((days_match, hours_match, minutes_match, seconds_match)):
        return None
    return timedelta(
        days=int(days_match.group(1)) if days_match else 0,
        hours=int(hours_match.group(1)) if hours_match else 0,
        minutes=int(minutes_match.group(1)) if minutes_match else 0,
        seconds=int(seconds_match.group(1)) if seconds_match else 0,
    )


def validate_value(value: Any) -> Union[str, int, float, timedelta]:
    _value = value
    if isinstance(value, str):
        _value = validate_string_type(value)  # type: str | int | float
    if isinstance(_value, (int, float)):
        return _value
    duration = parse_duration(str(value))
    if duration:
        return duration
    return _value


def parse_gcode_file(
    paths: Generator[Path, None, None]
) -> Dict[str, Dict[str, Any]]:
    result = {}  # type: Dict[str, Dict[str, Any]]

    # 값 추출을 위한 정규식 패턴을 정의합니다.
    for path in paths:
        with path.open() as file:
            # key 값을 추출합니다.
            key_match = KEY_PATTERN.search(str(path))
            if not key_match:
                continue
            key = key_match.group("key")
            values = {
                value_match.group("key"): validate_string_type(
                    value_match.group("value").strip()
                )
                for line in file.readlines()
                for value_match in VALUE_PATTERN.finditer(line)
            }
            result[key] = values
            # result[key] = {k: values.get(k, None) for k in KEYS_OF_INTEREST}

    return result


def plot_matrix(
    data_list: List[Dict[str, T]],
    x_label: str,
    y_label: str,
    value_label: str,
    value_extractor: Callable[[T], float],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    x_values = sorted(list(set([item[x_label] for item in data_list])))
    y_values = sorted(list(set([item[y_label] for item in data_list])))

    matrix = [
        [float("-inf") for _ in range(len(y_values))]
        for _ in range(len(x_values))
    ]

    for item in data_list:
        x_index = x_values.index(item[x_label])
        y_index = y_values.index(item[y_label])
        matrix[x_index][y_index] = value_extractor(item[value_label])

    matrix_np = np.array(matrix)
    matrix_normalized = (matrix_np - np.nanmin(matrix_np)) / (
        np.nanmax(matrix_np) - np.nanmin(matrix_np)
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix_normalized, cmap="Wistia")

    ax.set_xticks(range(len(x_values)))
    ax.set_yticks(range(len(y_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticklabels(y_values)

    for i in range(len(x_values)):
        for j in range(len(y_values)):
            value = matrix[i][j]
            value_str = (
                str(matrix[j][i]) if value != float("-inf") else "N/A"
            )  # noqa: E501
            ax.text(j, i, value_str, ha="center", va="center")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.colorbar(cax, orientation="vertical")
    plt.title(f"{value_label} Matrix")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None


if __name__ == "__main__":
    # gcode 파일을 읽어서 필요한 정보를 추출합니다.
    data = parse_gcode_file(Path("gcode").glob("*.gcode"))
    interested_data = {
        filename: {
            key: value
            for key, value in kv_dict.items()
            if key in KEYS_OF_INTEREST
        }
        for filename, kv_dict in data.items()
    }

    # 콘솔에 출력합니다.
    print(
        json.dumps(
            interested_data,
            indent=4,
        )
    )

    # 파일로 저장합니다.
    filename = f"result_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
    with open(filename, "w") as file:
        file.write(json.dumps(data, indent=4))
        print(f"파일 저장 완료: {filename}")

    data_list = [
        {**{"filename": filename}, **kv_dict}
        for filename, kv_dict in {
            filename: {
                key: validate_value(value) for key, value in kv_dict.items()
            }
            for filename, kv_dict in data.items()
        }.items()
    ]
    # data_list = [
    #     {"lt": 0.1, "is": 50, "td": timedelta(hours=1)},
    #     {"lt": 0.1, "is": 70, "td": timedelta(hours=2)},
    #     {"lt": 0.2, "is": 50, "td": timedelta(hours=3)},
    #     {"lt": 0.2, "is": 70, "td": timedelta(hours=4)},
    #     {"lt": 0.3, "is": 50, "td": timedelta(hours=5)},
    #     {"lt": 0.3, "is": 70, "td": timedelta(hours=6)},
    # ]
    plot_matrix(
        data_list,
        "first_layer_height",
        "infill_speed",
        KEYS_OF_INTEREST[3],
        lambda td: td.total_seconds(),
        Path("result.png"),
    )
