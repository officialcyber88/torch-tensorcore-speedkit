from __future__ import annotations
import pytest
from torch_speedkit.config import SpeedConfig


def test_config_defaults_validate():
    cfg = SpeedConfig()
    cfg.validate()


def test_invalid_precision():
    with pytest.raises(ValueError):
        SpeedConfig.from_dict({"precision": "fp8"})
