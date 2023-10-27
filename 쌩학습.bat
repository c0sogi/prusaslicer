set currentDir=%cd%
if exist .venv (
    echo .venv가 존재합니다. 파이썬 가상환경을 설정합니다.
    call .venv\Scripts\activate.bat
) else (
    echo .venv가 존재하지 않습니다. 파이썬 가상환경을 설정하지 않습니다.
)

set BASE_PATH=C:\Users\dcas\OneDrive\문서\카카오톡 받은 파일
set MODEL_PATH=%BASE_PATH%\ANN_E8707[LR=0.005][N1=40][N2=30][N3=20].keras

cd %currentDir%
call python -m training_ann --mode t1v --use_multiprocessing
call python -m training_ann --mode t12v --use_multiprocessing
call python -m training_ann --mode t123v --use_multiprocessing