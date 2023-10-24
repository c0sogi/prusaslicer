import pickle
import re
import pathlib

with open(r"C:\Users\sdml\Unghyeon\prusaslicer\output\MODE0\ANN_2023_10_24_142818.pkl", mode="rb") as f:
    pkl = pickle.load(f)

def extract_hyper_params_and_best_loss(data):
  hyper_params = data["train_input"]["hyper_params"]
  losses = data["train_output"]
  return hyper_params, min(losses["val_loss"])


N_BEST_MODELS = 5

hyper_params_and_losses = [extract_hyper_params_and_best_loss(data) for data in pkl]
sorted_hyper_params_and_losses = sorted(hyper_params_and_losses, key=lambda x: x[1])
best_models = sorted_hyper_params_and_losses[:N_BEST_MODELS]
print(best_models)

