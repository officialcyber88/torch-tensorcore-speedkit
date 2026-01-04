from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .config import SpeedConfig
from .attention import sdpa_backend_context
from .device import resolve_device
from .utils import set_seed
from .logging import setup_logger


log = setup_logger()


@dataclass
class SpeedContext:
    cfg: SpeedConfig
    device: torch.device
    autocast_dtype: Optional[torch.dtype]
    use_grad_scaler: bool


def _autocast_dtype_for(cfg: SpeedConfig) -> Optional[torch.dtype]:
    if not cfg.amp.enabled:
        return None
    if cfg.precision == "fp16":
        return torch.float16
    if cfg.precision == "bf16":
        return torch.bfloat16
    return None


def _set_tf32_matmul_precision(cfg: SpeedConfig) -> None:
    # Recommended control for TF32 is torch.set_float32_matmul_precision.
    # "high"/"medium" enable TF32 tensor cores for float32 matmuls on Ampere+.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(cfg.tf32_matmul_precision)


def _set_cudnn_flags(cfg: SpeedConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
        # deterministic is handled by set_seed(deterministic=True)
        # but we preserve user's explicit intent:
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def apply_speedups(model: torch.nn.Module, cfg: SpeedConfig) -> Tuple[torch.nn.Module, SpeedContext]:
    """
    Applies global backend knobs + optional torch.compile, returns (model, SpeedContext).
    """
    set_seed(cfg.seed, cfg.deterministic)
    dev = resolve_device(cfg.device)

    _set_tf32_matmul_precision(cfg)
    _set_cudnn_flags(cfg)

    model = model.to(dev)

    # Optional: channels_last for conv-heavy models (only affects 4D tensors).
    # We don't force it globally; we provide helper in examples.
    if cfg.channels_last and dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # torch.compile (Inductor) – speedups after first compilation run
    if cfg.compile.enabled:
        try:
            # Retrying call structure to avoid "Either mode or options" error
            compile_kwargs = {
                "backend": cfg.compile.backend,
                "fullgraph": cfg.compile.fullgraph,
                "dynamic": cfg.compile.dynamic,
            }
            if cfg.compile.options:
                compile_kwargs["options"] = dict(cfg.compile.options)
            else:
                compile_kwargs["mode"] = cfg.compile.mode

            model = torch.compile(model, **compile_kwargs)
            log.info(f"torch.compile enabled: backend={cfg.compile.backend} mode={cfg.compile.mode}")
        except Exception as e:
            log.warning(f"torch.compile failed, continuing eager. Reason: {type(e).__name__}: {e}")

    dtype = _autocast_dtype_for(cfg)
    use_scaler = bool(cfg.amp.enabled and cfg.amp.grad_scaler and cfg.precision == "fp16" and dev.type == "cuda")

    ctx = SpeedContext(cfg=cfg, device=dev, autocast_dtype=dtype, use_grad_scaler=use_scaler)
    return model, ctx


def autocast_context(ctx: SpeedContext):
    if not ctx.cfg.amp.enabled or ctx.autocast_dtype is None:
        # no-op context manager
        return torch.autocast(device_type=ctx.device.type, enabled=False)

    return torch.autocast(
        device_type=ctx.device.type,
        dtype=ctx.autocast_dtype,
        enabled=True,
        cache_enabled=bool(ctx.cfg.amp.autocast_cache_enabled),
    )


def sdp_context(ctx: SpeedContext):
    s = ctx.cfg.sdp
    return sdpa_backend_context(
        enable_flash=s.enable_flash,
        enable_mem_efficient=s.enable_mem_efficient,
        enable_math=s.enable_math,
        enable_cudnn=s.enable_cudnn,
    )
