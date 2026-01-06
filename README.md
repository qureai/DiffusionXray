# CheX-ray Super-Resolution (MUNIT-LQ → DDPM-HQ)

This repo contains code for our CheX-ray14 super-resolution experiments:

- **MUNIT-LQ**: low-quality (LQ) inputs generated from the CheX-ray14 test set (external MUNIT pipeline).
- **DDPM-HQ**: a diffusion-based super-resolution model (CheffSR/SR3-style) trained / fine-tuned on **HQ–LQ** pairs to map **LQ → HQ**.
- **DDPM-HQ outputs**: final super-res outputs produced by the DDPM-HQ model.

Most relevant entrypoints:
- **Inference**: `cheff/sr/sampler.py`, `inference_batch_ddp.py`
- **Training (DDP)**: `Train/train_ddpm_e2e.py`, `Train/train_sr_ddpm.py`

---

## Environment

We use the conda environment spec at `cascade7.yaml`:

```bash
conda env create -f cascade7.yaml
conda activate cascade7
```

---

## Where the data & outputs live (server paths)

### Outputs used in our CheX-ray14 runs

- **Diffusion-based baseline (bicubic SR)**: `/raid1/data_transfer/data/CheX-ray14/test_chexraybase_sr_bicubic/`
- **DDPM-HQ outputs (DDPM on LQ inputs)** (per-GPU folder): `/raid1/data_transfer/data/CheX-ray14/test_lr_ddpm_2/folder_x/`
- **MUNIT-LQ outputs**: `/raid2/data_transfer/data/CheX-ray14/test_lr_munit/all_images/all_images/`
- **DDPM-HQ outputs**: `/raid1/data_transfer/data/CheX-ray14/test_munit_sr_ddpm/`

### Curated qualitative bundle (100 matched images)

- Folder: `/raid/collected_100_4way/`
  - `diffusion_baseline/`
  - `ddpm_lq/`
  - `munit_lq/`
  - `final_with_masks/` (DDPM-HQ outputs)
  - `low_res_latest_160k/` (a 100-image low-res subset)
- Zip (4-way only): `/raid/collected_100_4way.zip`
- Zip (4-way + low-res): `/raid/collected_100_4way_with_lowres.zip`

---

## MUNIT-LQ (low-quality generation)

**Reference output path** (already generated on servers):

- `/raid2/data_transfer/data/CheX-ray14/test_lr_munit/all_images/all_images/`

Notes:
- Filenames are like `00000013_016.png` (no `super_resolved_` prefix).
- These images are used as **LQ inputs** to the DDPM-HQ SR model.

---

## DDPM-HQ training (super-resolution diffusion, HQ–LQ)

We train / fine-tune a CheffSR/SR3 diffusion model on **HQ–LQ** pairs to map **LQ → HQ**.

### Baseline SR3 training (bicubic-conditioned) on MaCheX-style data

Entrypoints:
- `Train/train_ddpm_e2e.py`
- `Train/scripts/04_train_chex_ddpm_ddp.py` (same training loop style)

Run (important: `diffusion.*` imports live under `Train/`, so run from `Train/`):

```bash
cd /raid1/data_transfer/data/chexray-diffusion/Train
PYTHONPATH=. python train_ddpm_e2e.py
```

Update paths inside `BASE_CONFIG`:
- `data_root`: MaCheX root (expects subfolders each containing an `index.json`)
- `resume_checkpoint`: optional checkpoint to resume from
- `log_dir`: where checkpoints are saved

### DDPM-HQ fine-tuning on our “low_res_latest” folder

Entrypoint:
- `Train/train_sr_ddpm.py`

By default it expects:
- `data_root/low_res_latest/` (PNG/JPG images)

In our setup, the latest low-res pool is here:
- `/raid2/data_transfer/low_res_latest_160k/`

If you want to use it without changing code, create a symlink named `low_res_latest` next to it:

```bash
ln -s /raid2/data_transfer/low_res_latest_160k /raid2/data_transfer/low_res_latest
```

Run:

```bash
cd /raid1/data_transfer/data/chexray-diffusion/Train
PYTHONPATH=. python train_sr_ddpm.py
```

Update paths inside `BASE_CONFIG` in `Train/train_sr_ddpm.py`:
- `data_root`: parent directory that contains `low_res_latest/`
- `resume_checkpoint`: base checkpoint to fine-tune from
- `log_dir`: output checkpoint folder

---

## Inference (DDPM-HQ)

### Multi-GPU batch inference on a folder (recommended)

Entrypoint:
- `inference_batch_ddp.py`

Edit the hardcoded variables at the bottom of the file:
- `sr_path`: checkpoint to load
- `input_folder`: folder containing input images (`.png` / `.jpg`)
- `output_folder`: where to write results
- `batch_size`: per-GPU batch size

Run:

```bash
cd /raid1/data_transfer/data/chexray-diffusion
python inference_batch_ddp.py
```

Outputs are saved as `super_resolved_<original_filename>` in `output_folder`.

Example (CheX-ray14 MUNIT-LQ → DDPM-HQ outputs):
- `input_folder`: `/raid2/data_transfer/data/CheX-ray14/test_lr_munit/all_images/all_images/`
- `output_folder`: `/raid1/data_transfer/data/CheX-ray14/test_lr_ddpm_2/folder_1/` (or any `folder_x/`)

### Programmatic inference helper (single GPU)

The core API is implemented in:
- `cheff/sr/sampler.py` (`CheffSRModel.sample_path(...)`, `CheffSRModel.sample_directory(...)`)

---

## Notes / gotchas

- **Training script imports**: `Train/train_*.py` imports `diffusion.*` from `Train/diffusion/`, so either:
  - run from `chexray-diffusion/Train` with `PYTHONPATH=.`, or
  - set `PYTHONPATH=/raid1/data_transfer/data/chexray-diffusion/Train`.
- `inference.py` is a minimal snippet and expects an in-memory `imgs` tensor; for folder inference prefer `inference_batch_ddp.py` or `CheffSRModel.sample_directory(...)`.
