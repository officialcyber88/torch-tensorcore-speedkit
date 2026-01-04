# SpeedKit Integration Guide

## What is SpeedKit?

**SpeedKit is NOT a trainer.** It's a **performance optimization layer** that speeds up your existing PyTorch training code.

### What It Does
- Automatically enables faster precision modes (TF32/BF16)
- Wraps your model to use Tensor Cores
- Leaves your training logic completely unchanged

### What YOU Still Do
- Define your model architecture
- Write your training loop
- Handle data loading
- Choose optimizer/loss function

---

## Quick Start: 3-Line Integration

```python
from torch_speedkit import SpeedConfig, apply_speedups

config = SpeedConfig.from_dict({"precision": "auto"})  # Auto-detects best settings
model, ctx = apply_speedups(model, config)

# In your training loop:
with ctx.autocast():
    output = model(input)
```

---

## Example 1: Basic Training Script

### BEFORE (Standard PyTorch)
```python
import torch
import torch.nn as nn

# Your model
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
```

### AFTER (With SpeedKit - Auto Mode)
```python
import torch
import torch.nn as nn
from torch_speedkit import SpeedConfig, apply_speedups

# Your model
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
).cuda()

# ✨ ADD SPEEDKIT (3 lines)
config = SpeedConfig.from_dict({"precision": "auto"})  # Auto-detects best settings
model, ctx = apply_speedups(model, config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop (only 1 line changed)
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        optimizer.zero_grad()
        with ctx.autocast():  # ✨ ADD THIS WRAPPER
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
```

---

## Example 2: Using Config File (Recommended)

### Step 1: Create `speedkit_config.yaml`
```yaml
precision: auto  # Automatically choose bf16/fp16/fp32
compile:
  enabled: auto  # Automatically enable/disable based on OS
tf32_matmul_precision: high
```

### Step 2: Update Your Training Script
```python
from torch_speedkit import SpeedConfig, apply_speedups

# Load config
config = SpeedConfig.from_yaml("speedkit_config.yaml")
model, ctx = apply_speedups(model, config)

# Rest of your training code unchanged!
for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    with ctx.autocast():
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
    loss.backward()
    optimizer.step()
```

---

## Example 3: Transformer/Hugging Face Models

```python
from transformers import BertForSequenceClassification
from torch_speedkit import SpeedConfig, apply_speedups

# Your Hugging Face model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased").cuda()

# Add SpeedKit
config = SpeedConfig.from_dict({"precision": "auto"})
model, ctx = apply_speedups(model, config)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    with ctx.autocast():  # Enable mixed precision
        outputs = model(**batch)
        loss = outputs.loss
    
    loss.backward()
    optimizer.step()
```

---

## Example 4: With Gradient Scaler (FP16)

For FP16 precision, you need gradient scaling for numerical stability:

```python
from torch_speedkit import SpeedConfig, apply_speedups

# Configure for fp16 (needs gradient scaling)
config = SpeedConfig.from_dict({
    "precision": "fp16",
    "grad_scaler": {"enabled": True}
})
model, ctx = apply_speedups(model, config)

# Get the scaler from context
scaler = ctx.grad_scaler

# Training loop
for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    
    with ctx.autocast():
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
    
    # Use scaler for fp16 stability
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Example 5: Custom Training Loop

```python
import torch
from torch_speedkit import SpeedConfig, apply_speedups

class MyTrainer:
    def __init__(self, model, config_path=None):
        # Apply SpeedKit
        if config_path:
            config = SpeedConfig.from_yaml(config_path)
        else:
            config = SpeedConfig.from_dict({"precision": "auto"})
        
        self.model, self.ctx = apply_speedups(model, config)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        
    def train_step(self, batch_x, batch_y):
        self.optimizer.zero_grad()
        
        # Use SpeedKit context
        with self.ctx.autocast():
            output = self.model(batch_x)
            loss = self.loss_fn(output, batch_y)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Usage
trainer = MyTrainer(my_model, config_path="speedkit_config.yaml")
for batch in dataloader:
    loss = trainer.train_step(batch_x, batch_y)
```

---

## Integration Checklist

Follow these steps to integrate SpeedKit into any training script:

### ✅ Step 1: Import
```python
from torch_speedkit import SpeedConfig, apply_speedups
```

### ✅ Step 2: Configure
Choose one:

**Option A: Auto mode (simplest)**
```python
config = SpeedConfig.from_dict({"precision": "auto"})
```

**Option B: YAML config (recommended)**
```python
config = SpeedConfig.from_yaml("speedkit_config.yaml")
```

**Option C: Manual settings**
```python
config = SpeedConfig.from_dict({
    "precision": "bf16",
    "tf32_matmul_precision": "high",
    "compile": {"enabled": False}
})
```

### ✅ Step 3: Apply to Model
```python
model, ctx = apply_speedups(model, config)
```

### ✅ Step 4: Wrap Forward Pass
```python
with ctx.autocast():
    output = model(input)
```

That's it! Your training will automatically use optimized precision.

---

## Auto-Detection Behavior

When you use `"auto"` mode, SpeedKit automatically:

| GPU | Precision | TF32 | Compile (Windows) |
|-----|-----------|------|-------------------|
| **Ada/Ampere+ (SM ≥ 8.0)** | bf16 | ✅ Enabled | ❌ Disabled |
| **Turing (SM 7.5)** | fp16 | ❌ N/A | ❌ Disabled |
| **Older GPUs** | fp32 | ❌ N/A | ❌ Disabled |
| **CPU** | fp32 | ❌ N/A | ❌ Disabled |

---

## When to Expect Speedups

✅ **Use SpeedKit when:**
- You have a PyTorch training script
- You want it to run faster on NVIDIA GPUs
- You have matmul-heavy models (transformers, large linear layers)
- Batch sizes are GPU-saturating (e.g., BS ≥ 256)

❌ **Don't expect speedups when:**
- Training on CPU only (gracefully degrades to fp32)
- Very small models/batches
- Conv-only models (smaller gains)

---

## Verification

After integrating SpeedKit, verify it's working:

### Check Your Hardware
```bash
python -m torch_speedkit.report --apply --restore
```

This shows:
- What settings SpeedKit auto-detected
- Before/after state of PyTorch flags
- Why each decision was made

### Run a Benchmark
```bash
python examples/benchmark_compare.py
```

This measures actual speedup on your hardware.

---

## Typical Integration Results

Based on verified benchmarks:

- **RTX 4050 Laptop (Ada)**: 1.41× speedup on Linear(8192, 8192) @ BS=512
- **CPU**: ~1.0× (no speedup, but no crashes either)
- **Transformers**: Expect 1.2-1.5× on Ampere+ GPUs
- **Small models**: May see < 1.1× (overhead dominates)

---

## Troubleshooting

### No Speedup?

1. **Check your GPU**:
   ```bash
   python -m torch_speedkit.report
   ```
   
2. **Verify TF32 is enabled**:
   ```python
   print(torch.backends.cuda.matmul.allow_tf32)  # Should be True after apply_speedups
   ```

3. **Increase workload size**: Try larger batch sizes or model dimensions

### Runtime Errors?

- **On Windows**: Make sure `compile: {enabled: auto}` (auto-disables)
- **FP16 instability**: Switch to `bf16` or add gradient scaler
- **Import errors**: Run `pip install -e .` in SpeedKit directory

---

## Example Projects

See working examples in the `examples/` directory:

- `train_toy_transformer.py` - Simple transformer training
- `train_cifar10_resnet.py` - CNN training (requires torchvision)
- `benchmark_compare.py` - Speed comparison tool

Run with:
```bash
python examples/train_toy_transformer.py --config examples/configs/auto_speed.yaml
```

---

## Summary

SpeedKit is like **adding a turbocharger to your car** - you still drive it the same way, but it goes faster.

It doesn't change WHAT you're training, just HOW FAST it trains.

**3 lines of code. Automatic speedups. No training logic changes.**
