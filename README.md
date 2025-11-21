# CSP: Causal Structure Preservation

Small research codebase for generating CSP synthetic datasets (T → M → Y* chain), training CARL models, and reproducing the benchmark tables included in `results/`.

## What’s here
- `csp_synth/`: synthetic data generator, SCM, MNIST-based imaging pipeline, packaging helpers.
- `cspsol/`: CARL model, training loop, evaluation hooks, metric calculators, CLI scripts.
- `experiments/`: default training config plus example checkpoints.
- `results/`: CSV outputs used in the paper (e.g., `results/table5_style_ib.csv`).

## Setup
1) Install Python 3.9+ and create a virtual environment.
2) Install PyTorch/torchvision matching your CUDA/CPU setup from https://pytorch.org/get-started/locally/.
3) Install the rest of the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick start: train on the bundled CSP-MNIST data
A ready-to-use dataset lives at `csp_synth/CSP-MNIST/cfg_dual_42`. Train a CARL model on the IM scenario:
```bash
python cspsol/scripts/train.py \
  --config experiments/carl_experiment/config.yaml \
  --data-dir csp_synth/CSP-MNIST/cfg_dual_42 \
  --scenario IM \
  --experiment-name demo_run \
  --output-dir experiments/demo_run_out
```
Checkpoints, logs, and `training_history.json` are written under `--output-dir`.

## Evaluate a checkpoint
```bash
python cspsol/scripts/eval.py \
  --checkpoint experiments/demo_run_out/demo_run.ckpt \
  --data-dir csp_synth/CSP-MNIST/cfg_dual_42 \
  --split test \
  --output-dir evaluation_results/demo_run
```

## Generate a new synthetic dataset
```bash
python csp_synth/scripts/gen_dataset.py csp_synth/configs/mnist_dual.yaml \
  --output-override csp_synth/generated/n5000_sigma0p3_neural \
  --seed-override 42
```
Flags such as `--im-only` or `--iy-only` restrict which image branch is produced; `--dry-run` prints the resolved configuration without writing files.

## Reproduce benchmark tables
Run the automated sweeps (defaults to 10 epochs per cell) and export fresh CSVs into `results/`:
```bash
python cspsol/scripts/run_tables.py --tables table5_style_ib --epochs 10
```
Add `--seeds 0 1 2` to aggregate multiple seeds (produces `*_seeded.csv`). Datasets are expected under `csp_synth/generated/` (the repo already contains several).
