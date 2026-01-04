from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from .config import SpeedConfig
from .speed import SpeedContext, autocast_context, sdp_context
from .optim import make_adamw
from .logging import setup_logger

log = setup_logger()


@dataclass
class StepMetrics:
    loss: float
    step_time_s: float
    lr: float


class Trainer:
    """
    Minimal, fast training loop that supports:
    - AMP autocast + GradScaler
    - gradient accumulation
    - optional grad clipping
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ctx: SpeedContext,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        make_batch: Callable[[int, torch.device], Tuple[torch.Tensor, torch.Tensor]],
    ):
        self.model = model
        self.ctx = ctx
        self.loss_fn = loss_fn
        self.make_batch = make_batch

        self.optimizer = make_adamw(
            self.model.parameters(),
            lr=ctx.cfg.optimizer.lr,
            weight_decay=ctx.cfg.optimizer.weight_decay,
            fused=ctx.cfg.optimizer.fused,
            foreach=ctx.cfg.optimizer.foreach,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=ctx.use_grad_scaler)

    def _lr(self) -> float:
        for g in self.optimizer.param_groups:
            return float(g["lr"])
        return 0.0

    def train(self) -> None:
        cfg = self.ctx.cfg
        self.model.train()

        total_steps = cfg.train.epochs * cfg.train.steps_per_epoch
        pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)

        global_step = 0
        for epoch in range(cfg.train.epochs):
            for step in range(cfg.train.steps_per_epoch):
                global_step += 1
                metrics = self.train_step()

                if (global_step % cfg.train.log_every) == 0:
                    pbar.set_postfix(
                        loss=f"{metrics.loss:.4f}",
                        step_time_ms=f"{metrics.step_time_s*1000.0:.1f}",
                        lr=f"{metrics.lr:.2e}",
                    )

                pbar.update(1)

        pbar.close()

    def train_step(self) -> StepMetrics:
        cfg = self.ctx.cfg
        t0 = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        accum = cfg.train.grad_accum_steps
        total_loss = 0.0

        # Use SDPA backend context (Flash/mem-efficient) if relevant to the model
        with sdp_context(self.ctx):
            for micro in range(accum):
                x, y = self.make_batch(cfg.train.batch_size, self.ctx.device)

                with autocast_context(self.ctx):
                    out = self.model(x)
                    loss = self.loss_fn(out, y) / float(accum)

                total_loss += float(loss.detach().item())

                if self.ctx.use_grad_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

        if cfg.train.clip_grad_norm is not None:
            max_norm = float(cfg.train.clip_grad_norm)
            if self.ctx.use_grad_scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

        if self.ctx.use_grad_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        t1 = time.perf_counter()
        return StepMetrics(loss=total_loss, step_time_s=(t1 - t0), lr=self._lr())
