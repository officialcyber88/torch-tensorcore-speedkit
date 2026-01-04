from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_speedkit.config import SpeedConfig
from torch_speedkit.speed import apply_speedups
from torch_speedkit.trainer import Trainer


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64, 64)
        self.l2 = torch.nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


def test_trainer_one_step_cpu():
    cfg = SpeedConfig.from_dict({
        "device": "cpu",
        "precision": "fp32",
        "compile": {"enabled": False},
        "train": {"epochs": 1, "steps_per_epoch": 1, "batch_size": 16, "grad_accum_steps": 1, "log_every": 1},
    })

    model = TinyNet()
    model, ctx = apply_speedups(model, cfg)

    def make_batch(bs: int, device: torch.device):
        x = torch.randn(bs, 64, device=device)
        y = torch.randint(0, 10, (bs,), device=device)
        return x, y

    def loss_fn(logits: torch.Tensor, y: torch.Tensor):
        return F.cross_entropy(logits, y)

    tr = Trainer(model, ctx, loss_fn, make_batch)
    tr.train()
