set currentDir=%cd%
if exist .venv (
    echo .venv�� �����մϴ�. ���̽� ����ȯ���� �����մϴ�.
    call .venv\Scripts\activate.bat
) else (
    echo .venv�� �������� �ʽ��ϴ�. ���̽� ����ȯ���� �������� �ʽ��ϴ�.
)

set BASE_PATH=C:\Users\dcas\OneDrive\����\īī���� ���� ����
set MODEL_PATH=%BASE_PATH%\ANN_E8707[LR=0.005][N1=40][N2=30][N3=20].keras

cd %currentDir%
call python -m training_ann --mode t1v --use_multiprocessing
call python -m training_ann --mode t12v --use_multiprocessing
call python -m training_ann --mode t123v --use_multiprocessing