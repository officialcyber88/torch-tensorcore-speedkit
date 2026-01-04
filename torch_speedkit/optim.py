from __future__ import annotations
from typing import Optional
import torch


def make_adamw(
    params,
    lr: float,
    weight_decay: float,
    fused: bool,
    foreach: Optional[bool],
) -> torch.optim.Optimizer:
    """
    AdamW with optional fused=True.
    If fused isn't supported on your build/device, PyTorch will raise; we fall back gracefully.
    """
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    if foreach is not None:
        kwargs["foreach"] = foreach

    if fused:
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except TypeError:
            # Older torch versions may not expose fused kwarg
            return torch.optim.AdamW(params, **kwargs)
        except RuntimeError:
            # Some builds/devices may not support fused; fall back.
            return torch.optim.AdamW(params, **kwargs)

    return torch.optim.AdamW(params, **kwargs)
