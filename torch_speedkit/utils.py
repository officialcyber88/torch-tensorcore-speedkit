from __future__ import annotations
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Determinism can reduce performance.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def maybe_channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    if x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def env_flag(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None and v != "" else None
