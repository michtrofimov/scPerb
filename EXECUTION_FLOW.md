# Execution Flow: `python scperb.py`

This document explains step-by-step what happens when you run `python scperb.py`.

## Overview

The script follows this main flow:
1. **Initialize configuration** → Parse arguments, detect device, set paths
2. **Load data** → Read `.h5ad` files, balance datasets
3. **Train model** → Train scPerb VAE for N epochs
4. **Generate predictions** → Save predictions to `.h5ad` files

---

## Step-by-Step Execution

### 1. Entry Point (`scperb.py` lines 123-139)

```python
if __name__ == '__main__':
    Opt = options()
    opt = Opt.init()
    print(opt)
    fix_seed(opt)
    dataset = customDataloader(opt)
    
    if opt.download_data == True:
        command = "python3 DataDownloader.py"
        subprocess.call([command], shell=True)
    
    if opt.validation == True:
        get_res(opt)
    else:
        train_model(opt, dataset)
        get_res(opt)
```

---

### 2. Configuration Initialization (`options/option.py`)

**`Opt.init()`** calls:

#### a) `get_opt()` - Parse command-line arguments
- Sets defaults (e.g., `data='pbmc'`, `device='cpu'`, `epochs=500`, `batch_size=256`)
- Creates argument parser with all hyperparameters

#### b) `make_dic()` - Create output directories
- Creates `save_path` (e.g., `scperb_saved/pbmc/`)
- Creates `model_save_path` (e.g., `scperb_saved/pbmc/model_scPerb/`)
- Creates `result_save_path` (e.g., `scperb_saved/pbmc/res_scPerb/`)

#### c) `check_device()` - Detect GPU/CPU
- Checks `torch.cuda.is_available()`
- If CUDA available → sets `opt.device = 'cuda'`
- Otherwise → keeps `opt.device = 'cpu'`
- **Now prints diagnostic info** (after our improvements)

#### d) `check_data()` - Set dataset-specific parameters
- Based on `opt.data` (pbmc/hpoly/salmonella/species/study):
  - Sets `stim_key`, `ctrl_key`, `cell_type_key`
  - Sets `input_dim` (number of genes, e.g., 6998 for pbmc)

**Result**: `opt` object with all configuration

---

### 3. Fix Random Seed (`scperb.py` lines 100-105)

```python
fix_seed(opt)
```
- Sets random seeds for reproducibility:
  - `random.seed(42)`
  - `torch.manual_seed(42)`
  - `np.random.seed(42)`
  - Enables deterministic CUDA operations

---

### 4. Data Loading (`dataloader/scperbDataset.py`)

**`dataset = customDataloader(opt)`** does:

#### a) Read training data
- Loads `train_pbmc.h5ad` (or other dataset) via `scanpy.read()`
- If `opt.supervise == False`:
  - Uses same file for train/valid
  - Excludes cells of type `opt.exclude_celltype` (default: 'CD4T')
- If `opt.supervise == True`:
  - Loads separate validation file
  - Filters to only `exclude_celltype` cells

#### b) Balance control/stimulated pairs
- `balance()` function:
  - Separates control (`ctrl_key`) and stimulated (`stim_key`) cells
  - For each cell type, balances to max(count_ctrl, count_stim)
  - Randomly samples to create balanced pairs
- Returns `con` (control) and `sti` (stimulated) AnnData objects

#### c) Convert to tensors
- Converts sparse matrices to dense numpy arrays
- Converts to PyTorch tensors
- Creates `sty` (style vector) - random tensor of size `input_dim`

#### d) Prepare validation data
- Extracts validation cells of `exclude_celltype`
- Separates control and stimulated validation data
- Prepares prediction input (control cells to predict stimulated)

**Result**: Dataset object ready for training

---

### 5. Optional: Download Data (`scperb.py` lines 130-132)

If `--download_data True`:
- Runs `DataDownloader.py` to fetch `.h5ad` files from Dropbox
- Downloads to `data/` directory

---

### 6. Training (`scperb.py` lines 65-98)

**`train_model(opt, dataset)`** executes:

#### a) Initialize model
```python
model = scperb(opt)  # Creates scPerb VAE model
model.to(opt.device)  # Moves to GPU/CPU
```

#### b) Optional resume
- If `opt.resume == True`:
  - Loads `*_now_epoch.pt` checkpoint

#### c) Create DataLoader
- PyTorch DataLoader with batch size, shuffling, pin_memory

#### d) Training loop (for each epoch 0 to `epochs-1`):

**For each batch:**
1. **Set input** (`model.set_input(con, sti, sty)`):
   - Moves control, stimulated, style tensors to device

2. **Forward pass** (`model.forward()`):
   - Passes through VAE encoder → latent space → decoder
   - Returns reconstructions and latent codes

3. **Compute loss** (`model.compute_loss(epoch)`):
   - **Reconstruction loss** (`rl`): SmoothL1Loss between predicted and actual stimulated
   - **KL divergence** (`kld_con`, `kld_sti`): Regularizes latent distributions
   - **Style loss** (`rl_sty`): Difference between control and stimulated latents
   - **Total loss**: `rl + 0.5 * alpha * kld_con + 0.5 * beta * kld_sti + rl_sty`

4. **Backward pass** (`model.backward()`):
   - `optimizer.zero_grad()`
   - `loss.backward()`
   - `optimizer.step()` (AdamW optimizer)

**After each epoch:**
- Saves model: `*_now_epoch.pt`
- Every 10 epochs: Runs validation
- Tracks best model based on validation score
- Updates progress bar with losses and scores

**Validation** (`validation()` function):
- Sets model to eval mode
- Predicts stimulated from control cells
- Computes R² scores (mean and variance)
- Returns composite score

---

### 7. Generate Final Predictions (`scperb.py` lines 107-121)

**`get_res(opt)`** runs after training:

#### a) Load best model
- Loads `*_best_epoch.pt` checkpoint

#### b) Run prediction
- Calls `validation(opt, model, get_result=True)`
- Predicts stimulated expression for all validation control cells
- Clips negative values to 0

#### c) Save results
- Creates AnnData object with predictions
- Saves to `result_save_path/best_epoch.h5ad`
- Also saves `now_epoch.h5ad` (last epoch, not best)

---

## Output Files

After execution, you'll have:

```
scperb_saved/pbmc/
├── model_scPerb/
│   ├── CD4T_best_epoch.pt      # Best model weights
│   └── CD4T_now_epoch.pt       # Latest model weights
└── res_scPerb/
    ├── best_epoch.h5ad          # Predictions from best model
    └── now_epoch.h5ad           # Predictions from latest model
```

---

## Key Parameters (Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | `'pbmc'` | Dataset name |
| `device` | `'cpu'` | Device (auto-detects CUDA) |
| `epochs` | `500` | Training epochs |
| `batch_size` | `256` | Batch size |
| `lr` | `1e-3` | Learning rate |
| `exclude_celltype` | `'CD4T'` | Cell type to predict |
| `alpha` | `0.01` | KL loss weight |
| `beta` | `0` | Additional KL weight |
| `delta` | `1` | Style loss weight |

---

## Command-Line Examples

```bash
# Basic training (default: pbmc dataset)
python scperb.py

# Use different dataset
python scperb.py --data hpoly

# Validation only (no training)
python scperb.py --validation True

# Resume training
python scperb.py --resume True

# Custom parameters
python scperb.py --epochs 1000 --batch_size 512 --lr 5e-4
```

---

## Notes

- **Device**: Automatically uses GPU if CUDA is available (check with `utils/check_cuda.py`)
- **Data**: Assumes `data/train_*.h5ad` and `data/valid_*.h5ad` exist
- **Memory**: Large datasets may require GPU or reduced batch size
- **Time**: Training 500 epochs can take hours depending on dataset size and hardware
