#!/usr/bin/env python3
"""Run CARL Table 4 experiment matrix (dataset/noise/nonlinearity sweep)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


# Ensure project root on sys.path when script executed directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cspsol.config.manager import ExperimentConfig


BASE_CONFIG = Path('experiments/carl_experiment/config.yaml')
OUTPUT_ROOT = Path('experiments/table4_runs')
RESULT_PATH = Path('results/table4.csv')
DATA_ROOT = Path('csp_synth/generated')
DEFAULT_EPOCHS = 10


@dataclass
class ExperimentSpec:
    name: str
    model_id: str
    overrides: Dict[str, object] = field(default_factory=dict)
    epochs: int = DEFAULT_EPOCHS
    data_dir: Optional[str] = None
    patch: Optional[Callable[[ExperimentConfig], None]] = None


def dataset_path(n_samples: int, noise: float, nonlinearity: str) -> Path:
    sigma_str = f"{noise:.2f}".rstrip('0').rstrip('.')
    sigma_slug = sigma_str.replace('.', 'p')
    return DATA_ROOT / f"n{n_samples}_sigma{sigma_slug}_{nonlinearity}"


def clone_config(base_cfg: ExperimentConfig) -> ExperimentConfig:
    serialized = json.loads(json.dumps(base_cfg, default=lambda o: getattr(o, '__dict__', o)))
    return ExperimentConfig.from_dict(serialized)


def prepare_config(base_cfg: ExperimentConfig, spec: ExperimentSpec, *, overwrite: bool = True) -> Path:
    cfg = clone_config(base_cfg)

    cfg.name = spec.name
    cfg.model.model_id = spec.model_id
    cfg.run.model_id = spec.model_id

    cfg.data.batch_size = getattr(base_cfg.data, 'batch_size', 16)
    cfg.device = 'cpu'
    cfg.training.max_epochs = spec.epochs
    cfg.training.use_amp = False

    for key, value in spec.overrides.items():
        section, attr = key.split('.')
        section_obj = getattr(cfg, section)
        setattr(section_obj, attr, value)

    if spec.patch is not None:
        spec.patch(cfg)

    out_dir = OUTPUT_ROOT / spec.name
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)

    cfg_path = out_dir / 'config.yaml'
    cfg.save(cfg_path)
    return cfg_path


def run_training(config_path: Path, spec: ExperimentSpec) -> None:
    cmd = [
        'python',
        'cspsol/scripts/train.py',
        '--config', str(config_path),
        '--experiment-name', spec.name,
        '--output-dir', str(OUTPUT_ROOT),
    ]
    if spec.data_dir:
        cmd.extend(['--data-dir', spec.data_dir])
    subprocess.run(cmd, check=True)


def extract_metrics(output_dir: Path) -> Dict[str, float]:
    history_path = output_dir / 'training_history.json'
    if not history_path.exists():
        raise FileNotFoundError(f'History file not found at {history_path}')
    data = json.loads(history_path.read_text())
    val = data['val_metrics']

    def get(metric: str) -> float:
        values = val.get(metric)
        if not values:
            return float('nan')
        return float(values[-1])

    return {
        'CSI': get('val_csi'),
        'MBRI': get('val_mbri'),
        'MAC': get('val_mac'),
        'Structural': get('val_struct_score'),
        'RIC-avg': get('val_ric_score'),
    }


def write_table(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def ensure_paths() -> ExperimentConfig:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return ExperimentConfig.load(BASE_CONFIG)


def format_metric(value: float) -> float:
    if isinstance(value, float):
        return round(value, 4)
    return float('nan')


def build_spec(n: int, sigma: float, nonlin: str, epochs: int) -> ExperimentSpec:
    data_dir_path = dataset_path(n, sigma, nonlin)
    if not data_dir_path.exists():
        raise FileNotFoundError(f'Dataset directory missing: {data_dir_path}')

    sigma_str = f"{sigma:.2f}".rstrip('0').rstrip('.')
    sigma_slug = sigma_str.replace('.', 'p')
    name = f"table4_n{n}_sigma{sigma_slug}_{nonlin}"

    def dataset_patch(cfg: ExperimentConfig) -> None:
        cfg.data.dataset_size = n
        cfg.data.noise_level = sigma
        cfg.data.nonlinearity = nonlin
        cfg.data.data_dir = str(data_dir_path)
        cfg.data.max_samples = n

    overrides = {
        'data.dataset_size': n,
    }

    return ExperimentSpec(
        name=name,
        model_id='carl_full',
        overrides=overrides,
        epochs=epochs,
        data_dir=str(data_dir_path),
        patch=dataset_patch,
    )


def run_table4(epochs: int, skip_existing: bool) -> None:
    base_cfg = ensure_paths()

    dataset_sizes = [500, 1000, 2000, 5000]
    noise_levels = [0.1, 0.3, 0.5]
    nonlinearities = ['linear', 'quadratic', 'neural']

    headers = ['数据规模', '噪声', '非线性', 'CSI', 'MBRI', 'MAC', 'Structural', 'RIC-avg']
    metric_keys = headers[3:]

    rows: List[List[object]] = []

    for n in dataset_sizes:
        for sigma in noise_levels:
            for nonlin in nonlinearities:
                spec = build_spec(n, sigma, nonlin, epochs)
                output_dir = OUTPUT_ROOT / spec.name
                history_path = output_dir / 'training_history.json'

                if skip_existing and history_path.exists():
                    metrics = extract_metrics(output_dir)
                else:
                    cfg_path = prepare_config(base_cfg, spec, overwrite=True)
                    run_training(cfg_path, spec)
                    metrics = extract_metrics(output_dir)

                row = [n, sigma, nonlin]
                row.extend(format_metric(metrics[key]) for key in metric_keys)
                rows.append(row)

    write_table(RESULT_PATH, headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CARL Table 4 experiment matrix only.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Epochs per setting (default: %(default)s)')
    parser.add_argument('--skip-existing', action='store_true', help='Reuse metrics if run directory already contains results')
    args = parser.parse_args()

    run_table4(args.epochs, args.skip_existing)


if __name__ == '__main__':
    main()
