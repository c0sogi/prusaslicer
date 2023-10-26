from pathlib import Path
from nn.dataloader import load_pickle

BASE_BEST = r"C:\Users\dcas\Documents\카카오톡 받은 파일\ANN_E8707[LR=0.005][N1=40][N2=30][N3=20].keras"
PETG_BEST = "output\\MODE2\\ANN_E9378[LR=0.005][N1=40][N2=30][N3=10].keras"
pkl = load_pickle(Path(BASE_BEST))
print(min(pkl["train_output"]["val_loss"]))
# 베이스 모델 val_loss 최소값: 6.8468%
# 베이스 모델 val_loss 최소값: 6.8468%
