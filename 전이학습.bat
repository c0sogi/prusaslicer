if exist .venv (
    echo .venv�� �����մϴ�. ���̽� ����ȯ���� �����մϴ�.
    call .venv\Scripts\activate.bat
) else (
    echo .venv�� �������� �ʽ��ϴ�. ���̽� ����ȯ���� �������� �ʽ��ϴ�.
)

set BASE_PATH=C:\Users\dcas\OneDrive\����\īī���� ���� ����
set MODEL_PATH=%BASE_PATH%\ANN_E8707[LR=0.005][N1=40][N2=30][N3=20].keras
set OUTPUT_PATH=./output
set COMMON_PARAMS=--model_path "%MODEL_PATH%" --lr 0.001 0.005 0.01 --patience 1000 --batch_size 2 4 8 12 --dropout_rate 0.0 0.25 0.5 --l1_reg 0.0 0.01 0.1

call python -m transfer_ann --mode a --train_indices 1 %COMMON_PARAMS% --output_path %OUTPUT_PATH%\mode_a_1
call python -m transfer_ann --mode a --train_indices 1 2 %COMMON_PARAMS% --output_path %OUTPUT_PATH%\mode_a_1_2
call python -m transfer_ann --mode a --train_indices 1 2 3 %COMMON_PARAMS% --output_path %OUTPUT_PATH%\mode_a_1_2_3
