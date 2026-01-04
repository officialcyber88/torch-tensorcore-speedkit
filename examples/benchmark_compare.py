
import argparse
import json
import platform
import datetime
import time
import warnings
import torch
import torch.nn as nn
from torch_speedkit.config import SpeedConfig
from torch_speedkit.speed import apply_speedups, autocast_context, sdp_context

# --- Helper ---
def dump_tf32_state(tag: str):
    """Print current TF32/precision state for auditing."""
    print(f"DEBUG: [{tag}] Matmul Precision: {torch.get_float32_matmul_precision()}")
    print(f"DEBUG: [{tag}] Matmul Allow TF32: {torch.backends.cuda.matmul.allow_tf32}")
    if torch.cuda.is_available():
        print(f"DEBUG: [{tag}] CuDNN Allow TF32: {torch.backends.cudnn.allow_tf32}")

# --- Model ---
class BenchModel(nn.Module):
    def __init__(self, dim=8192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)

def run_benchmark(label, config_path=None, use_speedkit=False, paranoid=False):
    print(f"\n--- Running: {label} ---")
    
    # 1. Setup
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Rigor: Ensure consistent CuDNN benchmark setting (False is safer for variable input sizes, though here consistent)
    torch.backends.cudnn.benchmark = False
    
    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    
    # Print version info for reproducibility (only once, in baseline)
    if not use_speedkit:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"CuDNN: {torch.backends.cudnn.version()}")
    
    dim = 8192
    bs = 512 # Optimized for 4050 Laptop speed
    
    # Print workload size for transparency
    print(f"Workload: Linear({dim}, {dim}) × 3 layers, batch_size={bs}")
    print(f"  → Main GEMM: ({bs}, {dim}) @ ({dim}, {dim}) [matmul-heavy]")
    
    model = BenchModel(dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = torch.randn(bs, dim, device=device)
    y = torch.randn(bs, dim, device=device)

    ctx = None
    if use_speedkit:
        if config_path is None:
            raise ValueError("Must provide config_path for SpeedKit run")
        
        # Snapshot: BEFORE SpeedKit
        dump_tf32_state("BEFORE SpeedKit")
        
        # Apply SpeedKit (suppress expected warnings in benchmark mode)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.compile disabled via auto-detection.*")
            cfg = SpeedConfig.from_yaml(config_path)
            cfg.device = device.type
            model, ctx = apply_speedups(model, cfg)
        
        # Snapshot: AFTER SpeedKit
        dump_tf32_state("AFTER SpeedKit")
        
        # We disabled fused optimizer in yaml, but let's stick to standard AdamW for now
        # to isolate the TF32 speedup.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    else:
        # Baseline: just show the default state
        dump_tf32_state("Baseline")
    
    # 2. Loop
    steps = 100
    warmup = 10
    
    # Helper for step execution
    def step_fn():
        optimizer.zero_grad(set_to_none=True)
        
        if use_speedkit and ctx:
            with sdp_context(ctx), autocast_context(ctx):
                out = model(x)
                loss = loss_fn(out, y)
            loss.backward() 
        else:
            # Baseline: "Normal" PyTorch
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            
        optimizer.step()
        
    model.train()
    
    # Warmup
    for _ in range(warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    # Timed
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    # Timed
    t0 = time.perf_counter()
    for _ in range(steps):
        if paranoid and device.type == "cuda":
            torch.cuda.synchronize()
        step_fn()
        if paranoid and device.type == "cuda":
            torch.cuda.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    avg_ms = (total_time / steps) * 1000.0
    print(f"Time: {avg_ms:.2f} ms/step")
    
    return {
        "label": label,
        "avg_ms": avg_ms,
        "std_dev_ms": 0.0, # Placeholder, simple timing
        "gpu_name": gpu_name,
        "config_path": config_path if use_speedkit else "N/A"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeedKit Benchmark & Verification Tool")
    parser.add_argument("--config", default="examples/configs/auto_speed.yaml", help="Path to SpeedKit config")
    parser.add_argument("--paranoid", action="store_true", help="Enable strict per-step synchronization")
    parser.add_argument("--json-export", help="Path to export benchmark results as JSON")
    args = parser.parse_args()

    # State isolation: capture initial global settings
    initial_matmul_prec = torch.get_float32_matmul_precision()
    initial_matmul_allow = torch.backends.cuda.matmul.allow_tf32
    initial_cudnn_allow = torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None

    # 1. Baseline (explicitly restore defaults first)
    torch.set_float32_matmul_precision(initial_matmul_prec)
    torch.backends.cuda.matmul.allow_tf32 = initial_matmul_allow
    if initial_cudnn_allow is not None:
        torch.backends.cudnn.allow_tf32 = initial_cudnn_allow
    
    base_stats = run_benchmark("Baseline (Standard PyTorch)", use_speedkit=False, paranoid=args.paranoid)
    base_ms = base_stats["avg_ms"]
    
    # Restore state between runs
    torch.set_float32_matmul_precision(initial_matmul_prec)
    torch.backends.cuda.matmul.allow_tf32 = initial_matmul_allow
    if initial_cudnn_allow is not None:
        torch.backends.cudnn.allow_tf32 = initial_cudnn_allow
    
    # 2. SpeedKit
    speed_label = f"SpeedKit (Auto-Detect: Windows / Ada SM 8.9)"
    speed_stats = run_benchmark(speed_label, config_path=args.config, use_speedkit=True, paranoid=args.paranoid)
    speed_ms = speed_stats["avg_ms"]
    
    # Restore state after runs
    torch.set_float32_matmul_precision(initial_matmul_prec)
    torch.backends.cuda.matmul.allow_tf32 = initial_matmul_allow
    if initial_cudnn_allow is not None:
        torch.backends.cudnn.allow_tf32 = initial_cudnn_allow
    
    # 3. Report
    ratio = base_ms / speed_ms
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Baseline:  {base_ms:.2f} ms")
    print(f"SpeedKit:  {speed_ms:.2f} ms")
    print(f"Speedup:   {ratio:.2f}x")
    print("="*40)

    # 4. JSON Export (Scientific Polish)
    if args.json_export:
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": speed_stats["gpu_name"],
            "paranoid": args.paranoid,
            "results": {
                "baseline": base_stats,
                "speedkit": speed_stats,
            },
            "speedup_ratio": ratio
        }
        with open(args.json_export, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Benchmark report saved to: {args.json_export}")
