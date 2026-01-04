import unittest
from unittest.mock import patch, MagicMock
from torch_speedkit.auto_config import get_device_capabilities, optimize_config

class TestAutoConfig(unittest.TestCase):
    @patch("sys.platform", "win32")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_windows_auto_detection(self, mock_props, mock_cuda):
        # Mock Ampere GPU on Windows
        mock_props.return_value = MagicMock(major=8, minor=6, name="RTX 3060")
        
        caps = get_device_capabilities()
        self.assertEqual(caps["os"], "win32")
        self.assertFalse(caps["supports_compile"])  # Windows -> False
        self.assertTrue(caps["supports_tf32"])
        self.assertEqual(caps["recommended_precision"], "bf16")

        # Test optimize_config
        cfg = {"compile": {"enabled": "auto"}, "precision": "auto"}
        optimized = optimize_config(cfg)
        self.assertFalse(optimized["compile"]["enabled"])
        self.assertEqual(optimized["precision"], "bf16")

    @patch("sys.platform", "linux")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_linux_pascal_detection(self, mock_props, mock_cuda):
        # Mock Pascal GPU (GTX 1080) on Linux
        mock_props.return_value = MagicMock(major=6, minor=1, name="GTX 1080")
        
        caps = get_device_capabilities()
        self.assertEqual(caps["os"], "linux")
        self.assertTrue(caps["supports_compile"])
        self.assertFalse(caps["supports_tf32"])
        self.assertEqual(caps["recommended_precision"], "fp32")
        
        # Test optimize_config
        cfg = {"compile": {"enabled": "auto"}, "precision": "auto"}
        optimized = optimize_config(cfg)
        self.assertTrue(optimized["compile"]["enabled"])
        self.assertEqual(optimized["precision"], "fp32")

if __name__ == "__main__":
    unittest.main()
