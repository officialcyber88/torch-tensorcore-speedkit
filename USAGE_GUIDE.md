# Torch TensorCore Speedkit

Torch TensorCore Speedkit is a high-performance wrapper that enables NVIDIA GPU optimizations in PyTorch with minimal boilerplate. Configure your preferred settings once, then apply them to your model and training loop.

---

## What it enables

| Optimization | What it does | Typical benefit |
|---|---|---|
| Automatic Mixed Precision (AMP) | Uses FP16/BF16 on supported ops (Tensor Cores) | Higher throughput, lower memory |
| TensorFloat-32 (TF32) | Accelerates FP32 matmul on Ampere+ | Faster FP32-heavy workloads |
| `torch.compile` | Graph capture + operator fusion via TorchDynamo | Less Python overhead, faster kernels |
| Fused optimizers | Uses fused kernels (e.g., fused AdamW) when available | Faster optimizer step |
| Flash Attention / SDPA backend control | Selects the best Scaled Dot Product Attention backend | Faster attention, better efficiency |

---

## Prerequisites

### System requirements

- **OS:** Windows or Linux  
- **GPU:** NVIDIA GPU (Volta/Turing/Ampere/Hopper recommended)  
- **Python:** 3.9+

### Install PyTorch (CUDA build)

You need a **CUDA-enabled** PyTorch build.

- Official install selector: `https://pytorch.org/get-started/locally/`

Example (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Install Speedkit (editable)

From your project folder:

```bash
cd "D:\ML Apps\torch-tensorcore-speedkit"
pip install -e .
```

---

## Usage

### 1) Create a config file

All optimizations are controlled through YAML.

Create `config.yaml` (or start from `examples/configs/speed.yaml`):

```yaml
device: "cuda"
precision: "auto"             # "auto" (detects bf16/tf32 support), fp16, bf16, fp32
tf32_matmul_precision: "high" # Enables TF32 on Ampere+

compile:
  enabled: "auto"             # "auto" (disables on Windows, enables on Linux)
  mode: "max-autotune"        # Aggressive optimization mode

optimizer:
  name: "adamw"
  fused: true                 # Use fused kernels if supported
```

### 2) Apply speedups in Python

#### Step A: Setup and apply speedups

This loads the config, builds your model normally, then applies:

- move to GPU  
- TF32 settings  
- optional `torch.compile`

```python
from torch_speedkit import SpeedConfig, apply_speedups

# Load configuration
cfg = SpeedConfig.from_yaml("config.yaml")

# Initialize your model normally
model = MyModel()

# Apply speedups
model, ctx = apply_speedups(model, cfg)
```

#### Step B: Training loop

Use the provided context managers to enable AMP and select SDPA backends automatically.

```python
from torch_speedkit import autocast_context, sdp_context
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for input, target in dataloader:
    optimizer.zero_grad()

    # Enable AMP + attention backend settings
    with sdp_context(ctx), autocast_context(ctx):
        output = model(input)
        loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()
```

---

## Auto-Detection System

SpeedKit includes a smart `auto_config` module that automatically adapts settings to your hardware and OS to prevent crashes and maximize speed.

- **Cross-Platform Safety**: Automatically disables `torch.compile` on Windows (where it is often unsupported) while keeping it enabled on Linux.
- **Hardware Optimization**:
    - **Ada / Ampere+ (RTX 30/40s)**: Automatically selects `bf16` precision and enables `TF32`.
    - **Turing (RTX 20s)**: Defaults to `fp16`.
    - **Older GPUs**: Defaults to safe `fp32`.

To use it, simply set specific fields to `"auto"` in your config (or use defaults):

```yaml
precision: "auto"
compile:
  enabled: "auto"
```

### Advanced: Overrides
You can override config settings via environment variables (Useful for clusters):
- `SPEEDKIT_COMPILE=0` (Force disable)
- `SPEEDKIT_PRECISION=bf16`

### Check your system (Active Report)
Run the report tool to see defaults, recommendations, and **apply them verify they work**:

```bash
python -m torch_speedkit.report --apply
```

This will show:
1. Current PyTorch defaults (usually inefficient)
2. Auto-detection logic (e.g., "Ada/Ampere+ confirmed")
3. **The result of applying SpeedKit** (e.g., `Matmul Allow TF32: True`)

---

## Verify performance (Paranoid Mode)

Run the benchmark (configured for strict per-step synchronization):

```bash
python examples/benchmark_compare.py
```

### Notes on expected results

- Enabling `compile.enabled`
- Using `precision: bf16` (or `fp16`)
- Setting `tf32_matmul_precision: high`

…should increase throughput (for example, higher samples/sec) compared to default settings.

---

Verified on: NVIDIA RTX 4050 Laptop GPU (Windows / Ada Lovelace)

Key finding:
PyTorch defaults to full FP32 for matrix multiplications on this hardware, often missing out on Tensor Core acceleration. `torch.compile` is also unstable on Windows.

With SpeedKit Auto-Detection:
- `compile` automatically DISABLED (preventing crashes)
- `precision` automatically set to `bf16` + `TF32` enabled
- Large workload (batch=512) for saturation

**Benchmark Conditions**:
- Model: `Linear(8192, 8192)` × 3 layers
- Batch Size: 512
- Paranoid Mode: Enabled (strict per-step synchronization)
- Warmup: 10 iterations
- Timed: 100 iterations
- CuDNN Benchmark: Disabled (for fairness)

Benchmark:
Baseline: 183.57 ms/step
SpeedKit: 131.79 ms/step
Speedup:  **1.39×** (Synced & Paranoid Verified)

### When to Expect Speedups

Speedups are **largest** when:
- Workload is **matmul-heavy** (e.g., large linear layers, transformers)
- Batch size and model dimensions are **large enough to saturate the GPU** (e.g., BS=512, Dim=8192)
- Using **Ada/Ampere+ GPUs** (SM ≥ 8.0) with TF32/BF16 support

Smaller models or batches may show less gain, as memory/overhead dominates over compute.
