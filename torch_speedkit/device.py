from __future__ import annotations
import torch


def resolve_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA is not available, but config.device='cuda'."
            if torch.version.cuda is None:
                msg += (
                    "\n    Diagnostic: Your PyTorch installation is CPU-only (torch.version.cuda is None)."
                    "\n    Solution: Reinstall PyTorch with CUDA support."
                    "\n    Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                )
            raise RuntimeError(msg)
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device}")


def describe_device(dev: torch.device) -> str:
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        return f"cuda:{idx} ({name}), capability={cap}"
    return "cpu"
