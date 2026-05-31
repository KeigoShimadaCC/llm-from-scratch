import random

import numpy as np
import torch

from kgpt.seed import seed_everything


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    seed_everything(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    assert first == second
