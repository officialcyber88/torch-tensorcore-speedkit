# torch-tensorcore-speedkit

**Enables TF32/BF16 Tensor Core fast paths in PyTorch via safe auto-detection, with auditable, reversible flag application and reproducible benchmarks.**

A reproducible performance protocol packaged as code.

It focuses on **real** levers PyTorch exposes:
- AMP autocast + GradScaler (Tensor Core acceleration for FP16/BF16 training)
- TF32 for float32 matmuls (Ampere+)
- torch.compile (TorchDynamo + Inductor/Triton)
- fused AdamW (when supported)
- SDPA backend toggles (Flash / mem-efficient / math)

> Note: On NVIDIA, Tensor Cores are accessed through CUDA libraries under the hood; the point here is: you don't write CUDA.

## Install

### 1. Install PyTorch (CUDA-enabled)

For CUDA 11.8 (recommended for RTX 30/40 series):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For other CUDA versions, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

### 2. Install SpeedKit

```bash
git clone https://github.com/yourusername/torch-tensorcore-speedkit.git
cd torch-tensorcore-speedkit
pip install -e .
# or with vision examples:
pip install -e ".[vision]"
```

## Quick Verify

Get instant feedback on your hardware:
```bash
python -m torch_speedkit.report --apply --restore
```

Run the verified benchmark:
```bash
python examples/benchmark_compare.py
```

## Quick start (benchmark)

```bash
python -m torch_speedkit.bench --config examples/configs/speed.yaml
```

## Quick start (toy training)

```bash
python examples/train_toy_transformer.py --config examples/configs/speed.yaml
# or CIFAR10 ResNet if you have torchvision:
python examples/train_cifar10_resnet.py --config examples/configs/speed.yaml
```

## Config knobs

See `examples/configs/speed.yaml` and `src/torch_speedkit/config.py`.

Key settings:

* precision: fp16 | bf16 | fp32
* tf32_matmul_precision: highest | high | medium
* compile: on/off + mode + inductor options like `shape_padding`
* optimizer: adamw + fused
* sdp_backend: flash/mem_efficient/math toggles (beta in PyTorch)

## Verified Results

### Environment
- **GPU**: NVIDIA GeForce RTX 4050 Laptop (Ada Lovelace, SM 8.9)
- **OS**: Windows 11
- **PyTorch**: 2.1.0+cu118
- **CUDA**: 11.8
- **CuDNN**: 8700

### Benchmark
- **Workload**: `Linear(8192, 8192)` × 3 layers, batch size 512
- **Paranoid sync**: Enabled (strict per-step GPU synchronization)
- **CuDNN benchmark**: Disabled (for fairness)

| Configuration | ms/step | Speedup |
|---------------|---------|---------|
| Baseline (FP32, TF32 off) | 184.68 | 1.00× |
| SpeedKit (BF16, TF32 on)  | 131.08 | **1.41×** |

**State changes verified**:
- `torch.get_float32_matmul_precision()`: `highest` → `high`
- `torch.backends.cuda.matmul.allow_tf32`: `False` → `True`

### When to Expect Speedups

Speedups are **largest** when:
- Workload is **matmul-heavy** (e.g., transformers, large linear layers)
- Batch size and model dimensions are **GPU-saturating** (e.g., BS ≥ 256, dim ≥ 4096)
- Using **Ada/Ampere+ GPUs** (SM ≥ 8.0) with TF32/BF16 support

**Always benchmark on your specific model + batch size.** Smaller workloads may show less gain.
