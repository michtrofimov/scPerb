# GPU Usage Fixes

## Issues Found

The code had several issues preventing proper GPU usage:

1. **Model loading**: `torch.load()` didn't specify `map_location`, so checkpoints might load to CPU even when GPU is available
2. **DataLoader**: `pin_memory=True` was always set, even for CPU (should only be True for CUDA)
3. **No verification**: No checks to confirm model/data are actually on GPU
4. **Missing optimizations**: `non_blocking=True` wasn't used for faster CPU→GPU transfers

## Fixes Applied

### 1. Model Loading (`models/scperb_model.py`)
```python
# Before:
self.model.load_state_dict(torch.load(path))

# After:
self.model.load_state_dict(torch.load(path, map_location=self.opt.device))
```
**Why**: Ensures checkpoint loads to the correct device (GPU if available).

### 2. Data Transfer (`models/scperb_model.py`)
```python
# Before:
self.con = con.to(self.opt.device)
self.sti = sti.to(self.opt.device)
self.sty = sty.to(self.opt.device)

# After:
self.con = con.to(self.opt.device, non_blocking=True)
self.sti = sti.to(self.opt.device, non_blocking=True)
self.sty = sty.to(self.opt.device, non_blocking=True)
```
**Why**: `non_blocking=True` allows async CPU→GPU transfers (faster when using `pin_memory`).

### 3. DataLoader Configuration (`scperb.py`)
```python
# Before:
dataloader = torch.utils.data.DataLoader(..., pin_memory=True)

# After:
pin_memory = (opt.device != 'cpu')
dataloader = torch.utils.data.DataLoader(..., pin_memory=pin_memory, num_workers=opt.num_workers)
```
**Why**: `pin_memory` only helps with GPU transfers. Also explicitly sets `num_workers`.

### 4. Model Initialization (`models/scperb_model.py`)
```python
# Added device verification:
if opt.device != 'cpu':
    first_param_device = next(self.model.parameters()).device
    print(f"  Model initialized on device: {first_param_device}")
```
**Why**: Verifies model is actually on GPU after initialization.

### 5. Device Verification (`scperb.py`)
Added debug output to show:
- Device configuration
- CUDA availability
- Model device after creation
- Model device after loading checkpoints

## Testing

Run the test script to verify GPU usage:

```bash
python3 test_gpu_usage.py
```

This will:
1. Check CUDA availability
2. Create model and verify it's on GPU
3. Move dummy data to GPU
4. Run forward pass
5. Show CUDA memory usage

## Expected Output

When running `python scperb.py`, you should now see:

```
✓ CUDA available: Using GPU (device: cuda)
  CUDA device count: 1
  Current CUDA device: 0
  CUDA device name: NVIDIA GeForce RTX 3090

Device Configuration:
  opt.device = 'cuda'
  torch.cuda.is_available() = True
  CUDA device: NVIDIA GeForce RTX 3090

  Model initialized on device: cuda:0
Model device check: cuda:0
```

## Monitoring GPU Usage

While training, monitor GPU usage with:

```bash
# In another terminal:
watch -n 1 nvidia-smi

# Or:
nvidia-smi -l 1
```

You should see:
- GPU memory usage increasing
- GPU utilization > 0%
- Memory allocated to your Python process

## Troubleshooting

If GPU still not used:

1. **Check CUDA installation**:
   ```bash
   python3 utils/check_cuda.py
   ```

2. **Verify device string**:
   - Should be `'cuda'` or `'cuda:0'` (not `'cpu'`)
   - Check output: `opt.device = 'cuda'`

3. **Check model placement**:
   - Look for: `Model device check: cuda:0`
   - If shows `cpu`, model didn't move to GPU

4. **Check data placement**:
   - Add print in training loop: `print(f"Batch device: {con.device}")`
   - Should show `cuda:0`, not `cpu`

5. **Force GPU**:
   ```bash
   python scperb.py --device cuda
   ```

## Notes

- The model is moved to device in `__init__` (line 19 of `scperb_model.py`)
- Data is moved to device in `set_input()` (called for each batch)
- `pin_memory=True` + `non_blocking=True` enables faster CPU→GPU transfers
- Model checkpoints are saved/loaded with device awareness
