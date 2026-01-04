from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from .config import SpeedConfig
from .speed import apply_speedups, autocast_context, sdp_context
from .logging import setup_logger

log = setup_logger()


@dataclass
class BenchResult:
    label: str
    step_ms: float
    throughput: float


def _warmup_steps() -> int:
    # Warmup helps stabilize GPU clocks and compilation caches.
    return 10


def _timed_steps() -> int:
    return 50


class TinyMLP(torch.nn.Module):
    def __init__(self, d_in: int = 4096, d_h: int = 4096, d_out: int = 4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_h),
            torch.nn.GELU(),
            torch.nn.Linear(d_h, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def benchmark(cfg: SpeedConfig) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    if cfg.device == "cuda" and dev.type != "cuda":
        raise RuntimeError("Config requests CUDA but CUDA isn't available.")

    torch.manual_seed(cfg.seed)

    model = TinyMLP()
    model, ctx = apply_speedups(model, cfg)

    # Synthetic data sized to push matmul kernels.
    bs = cfg.train.batch_size
    x = torch.randn(bs, 4096, device=ctx.device)
    y = torch.randn(bs, 4096, device=ctx.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
    loss_fn = torch.nn.MSELoss()

    # Warmup (also triggers compilation the first time if compile.enabled)
    model.train()
    for _ in range(_warmup_steps()):
        opt.zero_grad(set_to_none=True)
        with sdp_context(ctx), autocast_context(ctx):
            out = model(x)
            loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    if ctx.device.type == "cuda":
        torch.cuda.synchronize()

    # Timed
    iters = _timed_steps()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with sdp_context(ctx), autocast_context(ctx):
            out = model(x)
            loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    step_ms = (t1 - t0) * 1000.0 / float(iters)
    throughput = float(bs) / ((t1 - t0) / float(iters))

    log.info(f"Bench device={ctx.device.type} precision={cfg.precision} compile={cfg.compile.enabled}")
    log.info(f"Step time: {step_ms:.2f} ms | Throughput: {throughput:.1f} samples/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = SpeedConfig.from_yaml(args.config)
    benchmark(cfg)


if __name__ == "__main__":
    main()
