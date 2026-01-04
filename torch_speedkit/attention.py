from __future__ import annotations
import contextlib
import torch


@contextlib.contextmanager
def sdpa_backend_context(
    enable_flash: bool,
    enable_mem_efficient: bool,
    enable_math: bool,
    enable_cudnn: bool,
):
    """
    Controls which SDPA backends PyTorch is allowed to use.
    These flags are beta in PyTorch and may change.
    """
    if not torch.cuda.is_available():
        yield
        return

    # Save current
    prev = {}
    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        prev["flash"] = torch.backends.cuda.flash_sdp_enabled()
    if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
        prev["mem"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    if hasattr(torch.backends.cuda, "math_sdp_enabled"):
        prev["math"] = torch.backends.cuda.math_sdp_enabled()
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        prev["cudnn"] = torch.backends.cuda.cudnn_sdp_enabled()

    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(enable_flash)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(enable_math)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(enable_cudnn)
        yield
    finally:
        if "flash" in prev:
            torch.backends.cuda.enable_flash_sdp(prev["flash"])
        if "mem" in prev:
            torch.backends.cuda.enable_mem_efficient_sdp(prev["mem"])
        if "math" in prev:
            torch.backends.cuda.enable_math_sdp(prev["math"])
        if "cudnn" in prev:
            torch.backends.cuda.enable_cudnn_sdp(prev["cudnn"])
