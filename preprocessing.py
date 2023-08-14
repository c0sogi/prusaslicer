# flake8: noqa: E501
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import product
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class Settings:
    """Base class for settings."""

    @property
    def filename(self) -> str:
        raise NotImplementedError

    @property
    def ini_keys(self) -> Dict[str, Iterable[str]]:
        raise NotImplementedError


@dataclass
class FilamentSettings(Settings):
    fan_speed: int
    bed_temperature: int
    extruder_temperature: int
    infill_overlap: str

    @property
    def filename(self) -> str:
        return f"FS{self.fan_speed}_BT{self.bed_temperature}_ET{self.extruder_temperature}_IO{self.infill_overlap}.ini"

    @property
    def ini_keys(self) -> Dict[str, Iterable[str]]:
        return {
            "fan_speed": (
                "bridge_fan_speed",
                "fan_below_layer_time",
                "max_fan_speed",
                "min_fan_speed",
            ),
            "bed_temperature": (
                "first_layer_bed_temperature",
                "bed_temperature",
            ),
            "extruder_temperature": (
                "first_layer_temperature",
                "temperature",
            ),
            "infill_overlap": ("infill_overlap",),
        }


@dataclass
class PrintSettings(Settings):
    layer_thickness: float
    infill_speed: int

    bridge_speed: int = field(init=False)
    external_perimeter_speed: int = field(init=False)
    first_layer_speed: int = field(init=False)
    gap_fill_speed: int = field(init=False)
    ironing_speed: int = field(init=False)
    perimeter_speed: int = field(init=False)
    solid_infill_speed: int = field(init=False)
    top_solid_infill_speed: int = field(init=False)
    travel_speed: int = field(init=False, default=150)

    def __post_init__(self) -> None:
        self.bridge_speed = int(self.infill_speed * 1.2)
        self.external_perimeter_speed = int(self.infill_speed * 0.5)
        self.first_layer_speed = int(self.infill_speed * 0.6)
        self.gap_fill_speed = int(self.infill_speed * 0.4)
        self.ironing_speed = int(self.infill_speed * 0.3)
        self.perimeter_speed = int(self.infill_speed * 0.8)
        self.solid_infill_speed = int(self.infill_speed * 0.8)
        self.top_solid_infill_speed = int(self.infill_speed * 0.6)

    @property
    def filename(self) -> str:
        return f"LT{self.layer_thickness}_IS{self.infill_speed}.ini"

    @property
    def ini_keys(self) -> Dict[str, Iterable[str]]:
        return {
            "layer_thickness": (
                "first_layer_height",
                "layer_height",
            ),
            "infill_speed": ("infill_speed",),
            "bridge_speed": ("bridge_speed",),
            "external_perimeter_speed": ("external_perimeter_speed",),
            "first_layer_speed": (
                "first_layer_speed",
                "first_layer_speed_over_raft",
            ),
            "gap_fill_speed": ("gap_fill_speed",),
            "ironing_speed": ("ironing_speed",),
            "perimeter_speed": ("perimeter_speed",),
            "solid_infill_speed": ("solid_infill_speed",),
            "top_solid_infill_speed": ("top_solid_infill_speed",),
            "travel_speed": ("travel_speed",),
        }


def clean_folder(
    folder_name: str, except_files: Optional[List[str]] = ["template.ini"]
) -> None:
    for file_path in Path(folder_name).iterdir():
        if file_path.name.lower() not in (except_files or []):
            file_path.unlink()


def find_duplicate_keys(content: str) -> List[str]:
    keys = defaultdict(int)
    for line in content.split("\n"):
        if "=" in line:
            key = line.split("=", 1)[0].strip()
            keys[key] += 1
    return [key for key, count in keys.items() if count > 1]


def export_settings(
    settings: Settings,
    folder_name: str,
    template_filename: str = "template.ini",
) -> str:
    new_file_path = f"{folder_name}/{settings.filename}"

    # 새 파일을 작성하기 위해 template 파일 내용을 읽기
    with open(f"{folder_name}/{template_filename}", "r") as file:
        lines = file.readlines()

    # 설정을 변경하기 위한 딕셔너리 생성
    changes = {}
    for setting_key, setting_value in asdict(settings).items():
        for ini_key in settings.ini_keys[setting_key]:
            changes[ini_key] = str(setting_value)

    # 각 라인을 반복하며 필요한 변경을 적용
    for i, line in enumerate(lines):
        key = line.split("=")[0].strip()
        if key in changes:
            lines[i] = f"{key} = {changes[key]}\n"

    # 변경된 내용을 새 파일에 저장
    with open(new_file_path, "w") as file:
        file.writelines(lines)

    print(f"- {new_file_path} has been successfully created.")
    return new_file_path


def combine_configs(
    config: Dict[str, List[str]], folder_name: str
) -> List[str]:
    config_values: List[List[str]] = [config[key] for key in config]
    combinations: Tuple[Tuple[str]] = tuple(product(*config_values))
    print(f"- Total number of combinations: {len(combinations)}")

    combined_files: List[str] = []
    folder = Path(folder_name)
    for combination in combinations:
        combined_name = (
            "_".join([Path(item).stem for item in combination]) + ".ini"
        )
        combined_content = ""
        for file_path in combination:
            with open(file_path, "r") as file:
                combined_content += file.read() + "\n"

        duplicate_keys = find_duplicate_keys(combined_content)
        if duplicate_keys:
            lines = combined_content.split("\n")
            unique_content = ""
            processed_keys = set()
            for line in lines:
                if "=" in line:
                    key = line.split("=", 1)[0].strip()
                    if key in duplicate_keys:
                        if key not in processed_keys:
                            unique_content += line + "\n"
                            processed_keys.add(key)
                    else:
                        unique_content += line + "\n"
                else:
                    unique_content += line + "\n"
            combined_content = unique_content

        with open(folder / combined_name, "w") as file:
            file.write(combined_content)

        print(f"{combined_name} has been written.")
        combined_files.append(combined_name)
    return combined_files


# >>> PLA | ABS

# >>> 프린팅 속도
# min_print_speed = 15 | 10     -> 10으로 통일
# slowdown_below_layer_time = 20 | 15       -> 15으로 통일

# >>> 팬 속도  -> 10으로 통일
# bridge_fan_speed = 100 | 10
# fan_below_layer_time = 100 | 15
# max_fan_speed = 100 | 5
# min_fan_speed = 100 | 5

# >>> 베드 온도     -> 퍼스트 레이어 온도와 베드 온도를 같게 설정
# first_layer_bed_temperature = 55 | 80
# bed_temperature = 60 | 80

# >>> 압출 온도     -> 퍼스트 레이어 온도와 압출 온도를 같게 설정
# first_layer_temperature = 205 | 240
# temperature = 210 | 235
if __name__ == "__main__":
    # 프린터
    printer_path = "./printer/ultimaker2.ini"
    template_filename = "template.ini"
    config: Dict[str, List[str]] = {"printer": [printer_path]}

    # 필라멘트
    folder_name = "filament"
    fan_speeds = (0, 100)
    bed_temperatures = (50, 70)
    extruder_temperatures = (200, 210)
    infill_overlaps = ("15%", "35%")

    clean_folder(folder_name, except_files=[template_filename])
    config[folder_name] = []
    for setting in (
        FilamentSettings(
            fan_speed=fan_speed,
            bed_temperature=bed_temperature,
            extruder_temperature=extruder_temperature,
            infill_overlap=infill_overlap,
        )
        for fan_speed, bed_temperature, extruder_temperature, infill_overlap in product(
            fan_speeds,
            bed_temperatures,
            extruder_temperatures,
            infill_overlaps,
        )
    ):
        config[folder_name].append(
            export_settings(setting, folder_name, template_filename)
        )

    # 프린팅
    folder_name = "print"
    layer_thicknesses = (0.1, 0.2)
    infill_speeds = (50, 60)

    clean_folder(folder_name, except_files=[template_filename])
    config[folder_name] = []
    for setting in (
        PrintSettings(
            layer_thickness=layer_thickness,
            infill_speed=infill_speed,
        )
        for layer_thickness, infill_speed in product(
            layer_thicknesses, infill_speeds
        )
    ):
        config[folder_name].append(
            export_settings(setting, folder_name, template_filename)
        )

    # 최종 설정 파일 생성
    folder_name = "config"
    Path(folder_name).mkdir(exist_ok=True)
    clean_folder(folder_name, except_files=None)

    print(combine_configs(config, folder_name))
