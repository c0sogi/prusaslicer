set currentDir=%cd%
if exist .venv (
    echo .venv가 존재합니다. 파이썬 가상환경을 설정합니다.
    call .venv\Scripts\activate.bat
) else (
    echo .venv가 존재하지 않습니다. 파이썬 가상환경을 설정하지 않습니다.
)

set BASE_PATH=./output
set MODEL_PATH=%BASE_PATH%\ANN_E8707[LR=0.005][N1=40][N2=30][N3=20].keras
set OUTPUT_PATH=./output
set COMMON_PARAMS=--model_path "%MODEL_PATH%" --lr 0.001 0.005 0.01 --patience 1000 --batch_size 2 4 8 12 --use_multiprocessing

cd %currentDir%

call python -m transfer_ann --mode a --train_indices 1 2 %COMMON_PARAMS% --output_path %OUTPUT_PATH%\mode_a_1_2_f1 --freeze_layers dense_1

call python -m transfer_ann --mode a --train_indices 1 2 %COMMON_PARAMS% --output_path %OUTPUT_PATH%\mode_a_1_2_f12 --freeze_layers dense_1 dense_2

call python -m transfer_ann --mode a --train_indices 1 2 %COMMON_PARAMS% --output_path %OUTPUT_PATH%\mode_a_1_2_f123 --freeze_layers dense_1 dense_2 dense_3

