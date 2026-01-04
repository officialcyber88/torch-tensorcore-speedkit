from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
import os
import yaml
from .auto_config import optimize_config


Precision = Literal["fp16", "bf16", "fp32"]
MatmulPrecision = Literal["highest", "high", "medium"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
DeviceKind = Literal["cuda", "cpu"]


@dataclass
class AmpConfig:
    enabled: bool = True
    grad_scaler: bool = True
    autocast_cache_enabled: bool = True


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    fused: bool = True
    foreach: Optional[bool] = None


@dataclass
class CompileConfig:
    enabled: bool = True
    backend: str = "inductor"
    mode: CompileMode = "default"
    fullgraph: bool = False
    dynamic: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SdpConfig:
    enable_flash: bool = True
    enable_mem_efficient: bool = True
    enable_math: bool = True
    enable_cudnn: bool = True


@dataclass
class TrainLoopConfig:
    epochs: int = 1
    steps_per_epoch: int = 200
    batch_size: int = 32
    grad_accum_steps: int = 1
    clip_grad_norm: Optional[float] = 1.0
    log_every: int = 20


@dataclass
class SpeedConfig:
    seed: int = 1337

    device: DeviceKind = "cuda"
    precision: Precision = "bf16"

    tf32_matmul_precision: MatmulPrecision = "high"
    cudnn_benchmark: bool = True
    deterministic: bool = False

    channels_last: bool = False

    amp: AmpConfig = field(default_factory=AmpConfig)
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    sdp: SdpConfig = field(default_factory=SdpConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)

    @staticmethod
    def from_yaml(path: str) -> "SpeedConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return SpeedConfig.from_dict(data)


    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpeedConfig":
        # 1. Environment Variable Overrides
        # SPEEDKIT_PRECISION=bf16 overrides config
        env_precision = os.environ.get("SPEEDKIT_PRECISION", "")
        if env_precision:
            d["precision"] = env_precision

        env_compile = os.environ.get("SPEEDKIT_COMPILE", "")
        if env_compile:
            # support "0", "false", "off" as False
            if env_compile.lower() in ("0", "false", "off", "no"):
                d.setdefault("compile", {})["enabled"] = False
            elif env_compile.lower() in ("1", "true", "on", "yes"):
                d.setdefault("compile", {})["enabled"] = True
            elif env_compile == "auto":
                d.setdefault("compile", {})["enabled"] = "auto"

        # 2. Apply Auto-Detection Logic
        # This resolves keys that are set to "auto" (either from d or just now from env)
        d = optimize_config(d)

        def _merge(dc_cls, sub: Optional[Dict[str, Any]]):
            obj = dc_cls()
            if not sub:
                return obj
            for k, v in sub.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            return obj

        cfg = SpeedConfig()
        for k, v in d.items():
            if k in ("amp", "optimizer", "compile", "sdp", "train"):
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        cfg.amp = _merge(AmpConfig, d.get("amp"))
        cfg.optimizer = _merge(OptimConfig, d.get("optimizer"))
        cfg.compile = _merge(CompileConfig, d.get("compile"))
        cfg.sdp = _merge(SdpConfig, d.get("sdp"))
        cfg.train = _merge(TrainLoopConfig, d.get("train"))

        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            raise ValueError("device=cuda but CUDA_VISIBLE_DEVICES is empty.")

        if self.precision not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"Unknown precision: {self.precision}")

        if self.tf32_matmul_precision not in ("highest", "high", "medium"):
            raise ValueError(f"Unknown tf32_matmul_precision: {self.tf32_matmul_precision}")

        if self.train.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.train.epochs < 1 or self.train.steps_per_epoch < 1:
            raise ValueError("epochs and steps_per_epoch must be >= 1")
        if self.train.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.compile.enabled:
            if self.compile.mode not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
                raise ValueError(f"Invalid compile.mode: {self.compile.mode}")

        if self.optimizer.name.lower() != "adamw":
            raise ValueError("This speedkit currently supports optimizer.name=adamw only.")
