# Setup: Models on GitHub & Mamba Environments

## 1. Treating models in a GitHub repo

**Do not commit trained model weights (`.pt`, `.pth`, etc.) to the repo.** They are large, binary, and change often. This project already saves them under `saved/` and `*_saved*/`; these paths are in `.gitignore`.

### Options for sharing pretrained models

| Approach | Use when |
|----------|----------|
| **GitHub Releases** | Attach `.pt` (or a zip) to a release; link from README. Good for 1–2 small models. |
| **External storage** | Host on Zenodo, Hugging Face Hub, or Google Drive; document download in README or a script. Best for larger or many models. |
| **Git LFS** | Only if you must version few, small binaries in the repo. Requires `git lfs install` and can hit quota. |

### Recommended pattern

1. **Keep code and config in git** – model *code* in `models/`, paths in `options/option.py`.
2. **Ignore outputs** – `.gitignore` already excludes `saved/`, `*.pt`, and data files.
3. **Document how to get models** – e.g. in README: “Download pretrained weights from [Release / Zenodo / script] and place in `saved/...`” or “Train from scratch with `python scperb.py`.”
4. **Optional**: Add a small script (e.g. `scripts/download_pretrained.py`) that downloads weights to the paths your code expects.

---

## 2. Managing Mamba environments with `requirements.txt`

Mamba can create an env that uses your existing `requirements.txt` in two ways.

### Option A: Single `environment.yml` (recommended)

Use the provided `environment.yml`, which pins the env name and installs from `requirements.txt` via pip:

```bash
cd scPerb
mamba env create -f environment.yml
mamba activate scperb
```

To refresh the env after changing `requirements.txt`:

```bash
mamba env update -f environment.yml --prune
```

Or recreate from scratch:

```bash
mamba env remove -n scperb
mamba env create -f environment.yml
```

### Option B: Mamba + pip manually

Create the env, then install Python deps with pip:

```bash
mamba create -n scperb python=3.10 -y
mamba activate scperb
pip install -r requirements.txt
```

To export the current env back to `requirements.txt` (e.g. after adding packages):

```bash
pip freeze > requirements.txt
```

Or for a minimal list (only top-level deps):

```bash
pip install pip-tools
pip-compile requirements.in  # if you keep an .in file
# or
pip freeze | grep -v " @ " > requirements.txt
```

### Useful Mamba commands

| Task | Command |
|------|---------|
| Create from `environment.yml` | `mamba env create -f environment.yml` |
| Activate | `mamba activate scperb` |
| Update env from file | `mamba env update -f environment.yml --prune` |
| List envs | `mamba env list` |
| Remove env | `mamba env remove -n scperb` |
| Export env (conda packages only) | `mamba env export > environment.yml` |
| Export with pip deps | Use `environment.yml` with `pip: -r requirements.txt` (as in this repo) |

### Notes

- **Python version**: `environment.yml` uses Python 3.10; change the `python=3.10` line if you need another version.
- **CUDA**: For GPU, ensure the PyTorch from `requirements.txt` matches your CUDA version, or install PyTorch with conda/mamba before `pip install -r requirements.txt` and then comment out or remove the `torch*` lines from `requirements.txt` to avoid overwriting.
