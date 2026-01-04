import argparse
import torch
import sys
import warnings
from .auto_config import get_device_capabilities
from .config import SpeedConfig
from .speed import _set_tf32_matmul_precision, _set_cudnn_flags


def print_settings(title):
    print("-" * 60)
    print(f"{title}:")
    print(f"  * Matmul Precision:     {torch.get_float32_matmul_precision()}")
    print(f"  * Matmul Allow TF32:    {torch.backends.cuda.matmul.allow_tf32}")
    if torch.cuda.is_available():
        print(f"  * CuDNN Allow TF32:     {torch.backends.cudnn.allow_tf32}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply auto-detected SpeedKit settings to environment")
    parser.add_argument("--restore", action="store_true", help="Restore original settings after applying (for side-effect free checks)")
    args = parser.parse_args()

    print("="*60)
    print(" Torch SpeedKit Hardware Report")
    print("="*60)
    
    caps = get_device_capabilities()
    
    print(f"OS:                 {caps['os']}")
    print(f"CUDA Available:     {caps['cuda_available']}")
    
    if caps["cuda_available"]:
        print(f"GPU Name:           {caps['gpu_name']}")
        cc = caps['compute_capability']
        print(f"Compute Capability: {cc[0]}.{cc[1]} (TF32/BF16 supported)" if cc[0] >= 8 else f"{cc[0]}.{cc[1]}")
        
    print("-" * 60)
    print("Auto-Detection logic recommends:")
    print(f"  * Precision:       {caps['recommended_precision']}")
    print(f"  * Compile:         {caps['supports_compile']}")
    print(f"  * TF32:            {caps['supports_tf32']}")
    print(f"  * Decision Trace:  {caps['decision_trace']}")
    
    print_settings("Current PyTorch Defaults")

    if args.apply:
        # Capture old state
        old_matmul_prec = torch.get_float32_matmul_precision()
        old_matmul_allow = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_allow = torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None

        print("\n[Applying SpeedKit Auto-Configuration...]")
        # Simulate loading a config with "auto" defaults
        cfg_dict = {"precision": "auto", "compile": {"enabled": "auto"}}
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.compile disabled via auto-detection.*")
            cfg = SpeedConfig.from_dict(cfg_dict)
        
        # Apply global settings (TF32, CuDNN)
        _set_tf32_matmul_precision(cfg)
        _set_cudnn_flags(cfg)
        
        print_settings("After SpeedKit Application")
        
        print(f"Applicable Config Object:")
        print(f"  precision: {cfg.precision}")
        print(f"  compile:   {cfg.compile.enabled}")

        if args.restore:
            print("\n[Restoring Original Settings...]")
            torch.set_float32_matmul_precision(old_matmul_prec)
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_allow
            if old_cudnn_allow is not None:
                torch.backends.cudnn.allow_tf32 = old_cudnn_allow
            print_settings("Restored Settings")

    print("="*60)

if __name__ == "__main__":
    main()
