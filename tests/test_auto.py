import pytest
from unittest.mock import patch, MagicMock
from torch_speedkit.auto_config import get_device_capabilities, optimize_config

@patch("sys.platform", "win32")
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_properties")
def test_windows_auto_detection(mock_props, mock_cuda):
    # Mock Ampere GPU on Windows
    mock_props.return_value = MagicMock(major=8, minor=6, name="RTX 3060")
    
    caps = get_device_capabilities()
    assert caps["os"] == "win32"
    assert caps["supports_compile"] is False  # Windows -> False
    assert caps["supports_tf32"] is True
    assert caps["recommended_precision"] == "bf16"

    # Test optimize_config
    cfg = {"compile": {"enabled": "auto"}, "precision": "auto"}
    optimized = optimize_config(cfg)
    assert optimized["compile"]["enabled"] is False
    assert optimized["precision"] == "bf16"

@patch("sys.platform", "linux")
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_properties")
def test_linux_pascal_detection(mock_props, mock_cuda):
    # Mock Pascal GPU (GTX 1080) on Linux
    mock_props.return_value = MagicMock(major=6, minor=1, name="GTX 1080")
    
    caps = get_device_capabilities()
    assert caps["os"] == "linux"
    assert caps["supports_compile"] is True
    assert caps["supports_tf32"] is False
    assert caps["recommended_precision"] == "fp32"
    
    # Test optimize_config
    cfg = {"compile": {"enabled": "auto"}, "precision": "auto"}
    optimized = optimize_config(cfg)
    assert optimized["compile"]["enabled"] is True
    assert optimized["precision"] == "fp32"
