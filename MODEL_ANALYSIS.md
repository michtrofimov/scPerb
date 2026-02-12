# scPerb Model Parameter Count and GPU Memory Analysis

## Quick Summary

**For PBMC dataset checkpoint** (`CD4T_best_epoch.pt`):

- **Parameters**: ~13.4 million (most likely, without unused decoder2)
- **Model size**: ~51 MB
- **Training memory**: ~222 MB
- **Inference memory**: ~68 MB
- **Minimum GPU**: ~300 MB for training, ~90 MB for inference

**To get exact counts**, run: `python3 analyze_model.py` (requires PyTorch installed)

---

## Model Architecture

Based on `models/scperb_vae.py`, the scPerb model has the following architecture:

### For PBMC dataset:
- **Input dimension**: 6998 (genes)
- **Hidden dimension**: 800 (default)
- **Latent dimension**: 100 (default)

### Components:

1. **encoder1** (2-layer MLP):
   - Linear(input_dim → hidden_dim): 6998 × 800 = 5,598,400 params
   - BatchNorm1d(hidden_dim): 800 × 2 = 1,600 params (weight + bias)
   - Linear(hidden_dim → hidden_dim): 800 × 800 = 640,000 params
   - BatchNorm1d(hidden_dim): 800 × 2 = 1,600 params
   - **Subtotal**: ~6,241,600 params

2. **mu_encoder**:
   - Linear(hidden_dim → latent_dim): 800 × 100 = 80,000 params
   - **Subtotal**: 80,000 params

3. **logvar_encoder**:
   - Linear(hidden_dim → latent_dim): 800 × 100 = 80,000 params
   - **Subtotal**: 80,000 params

4. **encoder2** (style encoder):
   - Linear(input_dim → latent_dim): 6998 × 100 = 699,800 params
   - **Subtotal**: 699,800 params

5. **decoder** (3-layer MLP):
   - Linear(latent_dim → hidden_dim): 100 × 800 = 80,000 params
   - BatchNorm1d(hidden_dim): 800 × 2 = 1,600 params
   - Linear(hidden_dim → hidden_dim): 800 × 800 = 640,000 params
   - BatchNorm1d(hidden_dim): 800 × 2 = 1,600 params
   - Linear(hidden_dim → input_dim): 800 × 6998 = 5,598,400 params
   - **Subtotal**: ~6,321,600 params

6. **decoder2** (alternative decoder, same structure):
   - Same as decoder: ~6,321,600 params

### Total Parameters:

**If decoder2 is included**: ~19.8 million parameters
**If decoder2 is excluded**: ~13.5 million parameters

**Most likely**: ~13.5 million (decoder2 unused - see forward() method)

Note: Dropout layers don't have parameters, only BatchNorm and Linear layers do.

## Detailed Calculation

### encoder1:
- Linear(6998, 800): 6998 × 800 + 800 (bias) = 5,599,200
- BatchNorm1d(800): 800 (weight) + 800 (bias) = 1,600
- Linear(800, 800): 800 × 800 + 800 (bias) = 640,800
- BatchNorm1d(800): 800 + 800 = 1,600
- **Total encoder1**: 6,243,200

### mu_encoder:
- Linear(800, 100): 800 × 100 + 100 = 80,100

### logvar_encoder:
- Linear(800, 100): 800 × 100 + 100 = 80,100

### encoder2:
- Linear(6998, 100): 6998 × 100 + 100 = 699,900

### decoder:
- Linear(100, 800): 100 × 800 + 800 = 80,800
- BatchNorm1d(800): 1,600
- Linear(800, 800): 640,800
- BatchNorm1d(800): 1,600
- Linear(800, 6998): 800 × 6998 + 6998 = 5,605,398
- **Total decoder**: 6,330,198

### decoder2:
- Same as decoder: 6,330,198
- **Note**: decoder2 is NOT used in forward() or predict(), so may not be in checkpoint

### GRAND TOTAL:

**With decoder2** (if present):
6,243,200 + 80,100 + 80,100 + 699,900 + 6,330,198 + 6,330,198 = **19,763,696 parameters**

**Without decoder2** (most likely):
6,243,200 + 80,100 + 80,100 + 699,900 + 6,330,198 = **13,433,498 parameters**

## GPU Memory Usage

### Model Weights (float32):

**Scenario 1: With decoder2** (if present):
- Parameters: 19,763,696
- Memory: 19,763,696 × 4 bytes = **79,054,784 bytes ≈ 75.4 MB**

**Scenario 2: Without decoder2** (most likely):
- Parameters: 13,433,498
- Memory: 13,433,498 × 4 bytes = **53,733,992 bytes ≈ 51.2 MB**

### During Training:

**Without decoder2** (most likely):
1. **Model weights**: 51.2 MB
2. **Gradients**: 51.2 MB (same size as parameters)
3. **Optimizer states** (AdamW):
   - Momentum: 51.2 MB
   - Variance: 51.2 MB
   - **Total optimizer**: 102.4 MB
4. **Activations** (batch_size=256):
   - Input: 256 × 6998 × 4 bytes = 7.2 MB
   - Hidden layers: ~256 × 800 × 4 bytes × 3 = 2.5 MB
   - Latent: 256 × 100 × 4 bytes = 0.1 MB
   - Output: 256 × 6998 × 4 bytes = 7.2 MB
   - **Total activations**: ~17 MB (approximate)

**Total training memory**: 51.2 + 51.2 + 102.4 + 17 = **≈222 MB**

**With decoder2** (if present):
**Total training memory**: 75.4 + 75.4 + 150.8 + 17 = **≈318 MB**

### During Inference:

**Without decoder2**:
1. **Model weights**: 51.2 MB
2. **Activations**: ~17 MB
**Total inference memory**: 51.2 + 17 = **≈68 MB**

**With decoder2**:
**Total inference memory**: 75.4 + 17 = **≈92 MB**

## Summary

**Most Likely (without decoder2):**

| Metric | Value |
|--------|-------|
| **Total Parameters** | 13,433,498 (~13.4M) |
| **Model Size (weights)** | 51.2 MB |
| **Training Memory** | ~222 MB |
| **Inference Memory** | ~68 MB |
| **Minimum GPU (training)** | ~300 MB (~0.3 GB) |
| **Recommended GPU (training)** | ~450 MB (~0.45 GB) |
| **Minimum GPU (inference)** | ~90 MB (~0.09 GB) |

**If decoder2 is present:**

| Metric | Value |
|--------|-------|
| **Total Parameters** | 19,763,696 (~19.8M) |
| **Model Size (weights)** | 75.4 MB |
| **Training Memory** | ~318 MB |
| **Inference Memory** | ~92 MB |
| **Minimum GPU (training)** | ~400 MB (~0.4 GB) |
| **Recommended GPU (training)** | ~640 MB (~0.6 GB) |
| **Minimum GPU (inference)** | ~120 MB (~0.12 GB) |

## Running the Analysis Script

To get exact counts from your checkpoint:

```bash
cd scPerb
python3 analyze_model.py
```

This will:
1. Load the checkpoint
2. Count exact parameters
3. Estimate memory usage
4. Show detailed breakdown by layer

## Notes

- Actual memory may be higher due to PyTorch overhead
- Batch size affects activation memory (larger batch = more memory)
- Using mixed precision (float16) would halve memory usage
- The checkpoint file size may differ from model memory due to compression or optimizer states
