from pathlib import Path
import subprocess

from slice import PRUSA_CLI_EXECUTABLE


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
    for gcode_file in Path("gcode").glob("*.gcode"):
        cli = [
            PRUSA_CLI_EXECUTABLE,
            "--gcodeviewer",
            gcode_file.as_posix(),
        ]
        subprocess.run(cli)
