from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import List, Optional, Union

from config import STL_FILENAME

PRUSA_PATH = Path("C:/Program Files/Prusa3D/PrusaSlicer")
PRUSA_CLI_EXECUTABLE = (
    (PRUSA_PATH / "prusa-slicer-console.exe").as_posix()
    if PRUSA_PATH.exists()
    else "prusa-slicer-console.exe"
)


@dataclass
class SliceOptions:
    stl: Union[str, Path]
    config: Union[str, Path]
    output: Union[str, Path]

    def __post_init__(self):
        self.stl = Path(self.stl).resolve().as_posix()
        self.config = Path(self.config).resolve().as_posix()
        self.output = Path(self.output).resolve().as_posix()
        for file in (self.stl, self.config):
            if not Path(file).exists():
                raise ValueError(f"File {file} does not exist")

    @property
    def cli_args(self) -> List[str]:
        return [
            PRUSA_CLI_EXECUTABLE,
            "--load",
            str(self.config),
            "--export-gcode",
            str(self.stl),
            "--output",
            str(self.output),
        ]


def clean_folder(folder_name: str, except_files: Optional[List[str]] = None) -> None:
    for file_path in Path(folder_name).iterdir():
        if file_path.name.lower() not in (except_files or []):
            file_path.unlink()


def find_duplicate_keys(file_path: Union[str, Path]) -> List[str]:
    keys = {}
    duplicate_keys = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().strip("<").strip(">").strip()  # KEY의 공백 제거
                if key not in keys:
                    keys[key] = 1
                else:
                    keys[key] += 1

    for key, count in keys.items():
        if count > 1:
            duplicate_keys.append(key)
            print(f"중복된 키: {key}")

    return duplicate_keys


if __name__ == "__main__":
    try:
        exit_code = subprocess.call(
            [PRUSA_CLI_EXECUTABLE, "-h"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert exit_code == 0, "PrusaSlicer CLI 실행 불가"
    except Exception as e:
        print(f"PrusaSlicer CLI 실행 불가: {e}")
        raise e

    stl_path = Path(STL_FILENAME)
    output_folder_name = "gcode"
    clean_folder(output_folder_name)
    Path(output_folder_name).mkdir(exist_ok=True)
    config_file_names = sorted(Path("config").glob("*.ini"))
    for file_no, config_file in enumerate(config_file_names, start=1):
        duplicate_keys = find_duplicate_keys(config_file)
        if duplicate_keys:
            raise ValueError(f"Duplicate keys found in {config_file}: {duplicate_keys}")
        options = SliceOptions(
            stl=stl_path,
            config=config_file,
            output=Path(output_folder_name)
            / (f"{file_no}_" + config_file.name.replace(".ini", ".gcode")),
        )
        subprocess.run(options.cli_args, check=True, stdout=subprocess.DEVNULL)
