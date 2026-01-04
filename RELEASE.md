# torch-tensorcore-speedkit v0.1.0 Release Instructions

## Git Setup & Tag

```bash
cd "D:\ML Apps\torch-tensorcore-speedkit"

git init
git add .
git commit -m "Initial commit - v0.1.0"

# Ensure main branch name
git branch -M main

# Annotated tag
git tag -a v0.1.0 -m "torch-tensorcore-speedkit v0.1.0 - Auditable Tensor Core optimization with auto-detection"

# Add remote (replace with your real repo URL)
git remote add origin https://github.com/<yourusername>/torch-tensorcore-speedkit.git

# Push branch + tag
git push -u origin main
git push origin v0.1.0
```

## GitHub Release Notes Template

### torch-tensorcore-speedkit v0.1.0

Auditable PyTorch Tensor Core optimization via auto-detection.

---

#### What's New

##### Auto-Detection System
- Smart environment detection: OS (Windows/Linux) and GPU compute capability
- Platform-aware compile toggling (disables on Windows for stability)
- Hardware-aware precision selection:
  - Ada/Ampere+ (SM ≥ 8.0): TF32 + BF16
  - Turing (SM 7.5): FP16
  - Older/CPU: Safe FP32 fallback
- Decision tracing shows why each choice was made

##### Reporting & Verification Tools
- `torch_speedkit.report` CLI:
  - `--apply`: Demonstrates actual state changes
  - `--restore`: Reverts changes (CI-safe, no side effects)
- Benchmark harness with:
  - Before/after state snapshots (auditable)
  - Paranoid mode (strict per-step synchronization)
  - JSON export for machine-readable results
  - State isolation (repeatable across runs)

##### Verified Environment
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, SM 8.9)
- **OS**: Windows 11
- **PyTorch**: 2.1.0+cu118
- **CUDA**: 11.8
- **CuDNN**: 8700

##### Benchmark Results
- **Workload**: `Linear(8192, 8192)` × 3 layers, batch size 512
- **Baseline**: 185.16 ms/step (FP32, precision=highest, TF32 disabled)
- **SpeedKit**: 130.97 ms/step (BF16, precision=high, TF32 enabled)
- **Speedup**: **1.41×**
- **Configuration**: Paranoid sync enabled, CuDNN benchmark disabled

---

#### Quick Verify

Get instant feedback on your hardware:
```bash
python -m torch_speedkit.report --apply --restore
```

Run the verified benchmark:
```bash
python examples/benchmark_compare.py
```

---

#### Installation

```bash
# 1. Install PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install SpeedKit
git clone https://github.com/<yourusername>/torch-tensorcore-speedkit.git
cd torch-tensorcore-speedkit
pip install -e .
```

---

#### What This Does

Enables TF32/BF16 Tensor Core fast paths in PyTorch via safe auto-detection, with auditable, reversible flag application and reproducible benchmarks.

A reproducible performance protocol packaged as code.

---

See [CHANGELOG.md](CHANGELOG.md) for full details.
