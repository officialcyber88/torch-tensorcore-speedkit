from __future__ import annotations
import sys
import torch
import warnings
from typing import Dict, Any, Literal

def get_device_capabilities() -> Dict[str, Any]:
    """
    Detects hardware and OS capabilities to determine safe and optimal defaults.
    """
    capabilities = {
        "os": sys.platform,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "compute_capability": (0, 0),
        "supports_compile": True,
        "supports_tf32": False,
        "supports_bf16": False,
        "recommended_precision": "fp32", # Default safe fallback
        "decision_trace": [],
    }

    # 1. OS Checks
    if sys.platform == "win32":
        # torch.compile is generally not supported or unstable on Windows for PyTorch < 2.4
        # We disable it by default to prevent crashes.
        capabilities["supports_compile"] = False
        capabilities["decision_trace"].append("os=win32 -> compile disabled")
    elif sys.platform == "linux":
         capabilities["decision_trace"].append("os=linux -> compile enabled")

    # 2. GPU Checks
    if capabilities["cuda_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            capabilities["gpu_name"] = props.name
            capabilities["compute_capability"] = (props.major, props.minor)
            
            # Ampere (SM 8.0), Ada (SM 8.9), Hopper (SM 9.0) support TF32/BF16
            if props.major >= 8:
                capabilities["supports_tf32"] = True
                capabilities["supports_bf16"] = True
                capabilities["recommended_precision"] = "bf16" # Prefer BF16 on Ampere+
                capabilities["decision_trace"].append(f"sm={props.major}.{props.minor} (>=8.0) -> tf32/bf16 enabled")
            elif props.major == 7 and props.minor >= 5:
                # Turing (SM 7.5) supports FP16 Tensor Cores well
                capabilities["recommended_precision"] = "fp16"
                capabilities["decision_trace"].append("sm=7.5 (Turing) -> fp16 recommended")
            else:
                # Pascal (SM 6.1) or older
                capabilities["recommended_precision"] = "fp32"
                capabilities["decision_trace"].append(f"sm={props.major}.{props.minor} -> fp32 fallback")
                
        except Exception as e:
            warnings.warn(f"Failed to detect GPU capabilities: {e}")
            capabilities["decision_trace"].append(f"gpu_detection_failed: {e}")
    else:
        capabilities["decision_trace"].append("cuda_unavailable -> cpu defaults")

    return capabilities

def optimize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a raw config dictionary and applies auto-detected defaults where values are 'auto'.
    """
    caps = get_device_capabilities()
    
    # 1. Compile
    if config.get("compile", {}).get("enabled") == "auto":
        # If auto, enable only if supported
        config["compile"]["enabled"] = caps["supports_compile"]
        if not caps["supports_compile"]:
            warnings.warn("torch.compile disabled via auto-detection (Windows or unsupported OS detected).")
    elif config.get("compile", {}).get("enabled") is True and not caps["supports_compile"]:
         # Optional: Force disable if user enabled it but it will crash? 
         # For now, let's just warn to be safe, or just disable it if we are sure it crashes (like on Windows py2.1)
         if sys.platform == "win32":
             warnings.warn("torch.compile is enabled in config but may crash on Windows. Consider setting to 'auto' or 'false'.")

    # 2. Precision
    if config.get("precision") == "auto":
        config["precision"] = caps["recommended_precision"]
        
    # 3. TF32
    # If explicit config for tf32 is missing or auto, rely on capability
    # The config schema usually has specific strings like "high".
    # We can inject a check/warning if 'high' is set but hardware doesn't support it (Pascal).
    
    return config
