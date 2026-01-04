from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn.functional as F

from torch_speedkit.config import SpeedConfig
from torch_speedkit.speed import apply_speedups
from torch_speedkit.trainer import Trainer
from torch_speedkit.logging import setup_logger

log = setup_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = SpeedConfig.from_yaml(args.config)

    try:
        import torchvision
        import torchvision.transforms as T
    except Exception as e:
        raise RuntimeError(
            "torchvision not available. Install extras: pip install -e '.[vision]'"
        ) from e

    # Simple ResNet
    model = torchvision.models.resnet18(num_classes=10)

    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=True,
    )
    it = iter(dl)

    model, ctx = apply_speedups(model, cfg)

    def make_batch(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal it
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return x, y

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y)

    trainer = Trainer(model=model, ctx=ctx, loss_fn=loss_fn, make_batch=make_batch)
    log.info("Starting CIFAR10 training...")
    trainer.train()


if __name__ == "__main__":
    main()
