from pathlib import Path
import unittest

from nn.dataloader import read_ss_curves


class TestSSDataLoader(unittest.TestCase):
    def test_load(self):
        all_curves = read_ss_curves(Path("./raw_ssdata"))
        for case, curves in all_curves.items():
            print(f"===== Case {case} =====")
            for curve in curves:
                print(len(curve["curve"]["strain"]), end=" ")
            print()
