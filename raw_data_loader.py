from pathlib import Path

from nn.dataloader import read_all_ss_curves

if __name__ == "__main__":
    all_curves = read_all_ss_curves(Path("./raw_ssdata"))
    print(all_curves.keys())
