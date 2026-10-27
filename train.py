import warnings

import torch

from transplat.src.main import train


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    train()
