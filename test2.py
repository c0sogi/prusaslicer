import pickle
import pathlib

PKL_FILE_PATH = r"output\MODE2\ANN_2023_10_26_165137.pkl"


def find_files_with_pattern(params):
    # 패턴 생성
    pattern = "".join(
        [f"[{key.upper()}={value}]" for key, value in params.items()]
    )

    # 현재 폴더에서 .keras로 끝나는 모든 파일을 검사
    folder = pathlib.Path(PKL_FILE_PATH).parent
    matched_files = [
        str(file) for file in folder.glob("*.keras") if pattern in file.name
    ]

    return matched_files


with open(PKL_FILE_PATH, mode="rb") as f:
    pkl = pickle.load(f)


def extract_hyper_params_and_best_loss(data):
    hyper_params = data["train_input"]["hyper_params"]
    losses = data["train_output"]
    return hyper_params, min(losses["val_loss"])


N_BEST_MODELS = 5

hyper_params_and_losses = [
    extract_hyper_params_and_best_loss(data) for data in pkl
]
sorted_hyper_params_and_losses = sorted(
    hyper_params_and_losses, key=lambda x: x[1]
)
best_models = sorted_hyper_params_and_losses[:N_BEST_MODELS]
for i, (hyper_params, loss) in enumerate(best_models):
    print(f"No.{i+1}: {find_files_with_pattern(hyper_params)}")
