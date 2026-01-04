# Changelog

## [0.1.0] - 2026-01-04

### Added

#### Auto-Detection System
- **Smart environment detection**: Automatically detects OS (Windows/Linux) and GPU compute capability
- **Platform-aware compile toggling**: Disables `torch.compile` on Windows (where it's unstable in PyTorch < 2.4)
- **Hardware-aware precision selection**:
  - Ada/Ampere+ (SM ≥ 8.0): Enables TF32 + recommends BF16
  - Turing (SM 7.5): Recommends FP16
  - Older GPUs: Safe FP32 fallback
- **Configuration hierarchy**: User config > Environment variables > Auto-detected defaults
- **Decision tracing**: Logs why each configuration choice was made

#### Reporting & Verification
- **`torch_speedkit.report` CLI tool**:
  - Shows hardware capabilities and auto-detection decisions
  - `--apply` flag: Demonstrates actual state changes
  - `--restore` flag: Reverts changes (no side effects, CI-safe)
- **Benchmark harness** (`examples/benchmark_compare.py`):
  - Before/after state snapshots (auditable)
  - Paranoid mode: Strict per-step synchronization
  - JSON export for machine-readable results
  - Workload transparency (prints GEMM dimensions)
  - State isolation (repeatable across runs)

#### Testing
- Unit tests for auto-detection logic (mocked OS/GPU scenarios)
- Benchmark fairness controls (consistent CuDNN settings)

### Verified Environment
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, SM 8.9)
- **OS**: Windows 11
- **PyTorch**: 2.1.0+cu118
- **CUDA**: 11.8
- **CuDNN**: 8700

### Benchmark Results
- **Workload**: `Linear(8192, 8192)` × 3 layers, batch size 512
- **Baseline**: 184.68 ms/step (FP32, precision=highest, TF32 disabled)
- **SpeedKit**: 131.08 ms/step (BF16, precision=high, TF32 enabled)
- **Speedup**: **1.41×**
- **Configuration**: Paranoid sync enabled, CuDNN benchmark disabled

### Architecture
- Reversible state changes (all global PyTorch flags can be restored)
- Transparent decision-making (decision trace shows logic)
- No hidden side effects
