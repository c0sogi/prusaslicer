import os
import subprocess
import sys

# 현재 디렉터리 저장
currentDir = os.getcwd()

# 경로 및 파라미터 설정
BASE_PATH = "C:\\Users\\dcas\\OneDrive\\문서\\카카오톡 받은 파일"
MODEL_PATH = os.path.join(
    BASE_PATH, "ANN_E8707[LR=0.005][N1=40][N2=30][N3=20].keras"
)
OUTPUT_PATH = "./output"
COMMON_PARAMS = [
    "--model_path",
    MODEL_PATH,
    "--lr",
    "0.001",
    "0.005",
    "0.01",
    "--patience",
    "1000",
    "--batch_size",
    "2",
    "4",
    "8",
    "--dropout_rate",
    "0.0",
    "0.25",
    "0.5",
    "--l1_reg",
    "0.0",
    "0.01",
    "0.1",
]


# python 명령어 실행
def run_command(mode, train_indices, output_suffix):
    cmd = (
        [
            sys.executable,
            "-m",
            "transfer_ann",
            "--mode",
            mode,
            "--train_indices",
        ]
        + train_indices
        + COMMON_PARAMS
        + ["--output_path", os.path.join(OUTPUT_PATH, output_suffix)]
    )
    subprocess.call(cmd, cwd=currentDir)


run_command("a", ["1"], "mode_a_1")
run_command("a", ["1", "2"], "mode_a_1_2")
run_command("a", ["1", "2", "3"], "mode_a_1_2_3")