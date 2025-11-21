#!/usr/bin/env python3
"""Run CARL benchmark tables and aggregate metrics."""

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

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cspsol.config.manager import ExperimentConfig

BASE_CONFIG = Path('experiments/carl_experiment/config.yaml')
OUTPUT_ROOT = Path('experiments/table_runs')
RESULT_ROOT = Path('results')
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
    seed: Optional[int] = None


def dataset_path(n_samples: int, noise: float, nonlinearity: str = 'quadratic') -> Path:
    sigma_str = f"{noise:.2f}".rstrip('0').rstrip('.')
    sigma_slug = sigma_str.replace('.', 'p')
    return DATA_ROOT / f"n{n_samples}_sigma{sigma_slug}_{nonlinearity}"


def prepare_config(base_cfg: ExperimentConfig, spec: ExperimentSpec) -> Path:
    cfg = ExperimentConfig.from_dict(json.loads(json.dumps(base_cfg, default=lambda o: getattr(o, '__dict__', o))))

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
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cfg.output_dir = str(out_dir)

    if spec.seed is not None:
        cfg.random_seed = int(spec.seed)
        cfg.data.random_seed = int(spec.seed)

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
    if spec.seed is not None:
        cmd.extend(['--seed', str(spec.seed)])
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


def format_metric(value: float) -> float:
    if isinstance(value, float):
        return round(value, 4)
    return float('nan')


def ensure_paths() -> ExperimentConfig:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_cfg = ExperimentConfig.load(BASE_CONFIG)
    return base_cfg


def run_tables(epochs: int, tables: Optional[List[str]] = None, seeds: Optional[List[int]] = None, skip_existing: bool = False) -> None:
    base_cfg = ensure_paths()
    tables_set = set(tables) if tables else None
    seeds_list = list(seeds) if seeds else [None]

    def spec_with_dataset(name: str, model_id: str, n: int, sigma: float, nonlin: str, epochs_override: int = epochs, patch: Optional[Callable[[ExperimentConfig], None]] = None) -> ExperimentSpec:
        data_dir = dataset_path(n, sigma, nonlin)

        def dataset_patch(cfg: ExperimentConfig) -> None:
            cfg.data.dataset_size = n
            cfg.data.max_samples = n
            cfg.data.noise_level = sigma
            cfg.data.nonlinearity = nonlin
            cfg.data.data_dir = str(data_dir)
            if patch is not None:
                patch(cfg)

        overrides: Dict[str, object] = {
            'data.dataset_size': n,
        }
        return ExperimentSpec(name, model_id, overrides, epochs_override, data_dir=str(data_dir), patch=dataset_patch)

    def loss_toggle_patch(disabled: Optional[List[str]] = None, enabled_only: Optional[List[str]] = None) -> Callable[[ExperimentConfig], None]:
        disabled = disabled or []
        enabled_only = enabled_only or []

        def _patch(cfg: ExperimentConfig) -> None:
            loss_cfg = cfg.model.loss_config
            if enabled_only:
                for key in loss_cfg:
                    loss_cfg[key]['enabled'] = key in enabled_only
            for key in disabled:
                if key in loss_cfg:
                    loss_cfg[key]['enabled'] = False
        return _patch

    def enable_losses_patch(
        to_enable: List[str],
        weight_overrides: Optional[Dict[str, float]] = None,
        include_in_full_phase: bool = True,
    ) -> Callable[[ExperimentConfig], None]:
        """
        Explicitly enable a set of losses (useful when defaults are disabled).

        Args:
            to_enable: Loss names to switch on.
            weight_overrides: Optional weights to set for the enabled losses.
            include_in_full_phase: Whether to ensure the losses appear in the
                full-phase enabled list for reproducibility.
        """
        weight_overrides = weight_overrides or {}

        def _patch(cfg: ExperimentConfig) -> None:
            loss_cfg = cfg.model.loss_config
            for name in to_enable:
                # Create placeholder if it doesn't exist yet
                if name not in loss_cfg:
                    loss_cfg[name] = {}
                loss_cfg[name]['enabled'] = True
                if name in weight_overrides:
                    loss_cfg[name]['weight'] = weight_overrides[name]

            if include_in_full_phase:
                phase_cfg = cfg.model.phase_config
                full_cfg = phase_cfg.setdefault('full', {})
                enabled = full_cfg.setdefault('enabled_losses', [])
                for name in to_enable:
                    if name not in enabled:
                        enabled.append(name)

        return _patch

    def compact_phase_patch(warmup1_end: int = 2, warmup2_end: int = 4) -> Callable[[ExperimentConfig], None]:
        """
        Shorten phase ranges so optional losses (style/IB) become active within
        small epoch budgets used in automated table runs.
        """

        def _patch(cfg: ExperimentConfig) -> None:
            loss_cfg = cfg.model.loss_config

            def _is_enabled(name: str) -> bool:
                return loss_cfg.get(name, {}).get('enabled', False)

            warmup1_losses = []
            for cand in ['align', 'mac', 'ci', 'mbr']:
                if _is_enabled(cand):
                    warmup1_losses.append(cand)
            if not warmup1_losses:
                warmup1_losses = [name for name, lc in loss_cfg.items() if lc.get('enabled')]

            warmup2_losses = [name for name in ['ci', 'mbr', 'mac', 'align'] if _is_enabled(name)]
            full_losses = [name for name, lc in loss_cfg.items() if lc.get('enabled')]

            cfg.model.phase_config = {
                'warmup1': {
                    'epochs': [0, warmup1_end],
                    'enabled_losses': warmup1_losses,
                    'use_grl': False,
                    'use_vib': False,
                },
                'warmup2': {
                    'epochs': [warmup1_end, warmup2_end],
                    'enabled_losses': warmup2_losses,
                    'use_grl': False,
                    'use_vib': False,
                },
                'full': {
                    'epochs': [warmup2_end, float('inf')],
                    'enabled_losses': full_losses,
                    'use_grl': True,
                    'use_vib': True,
                },
            }

        return _patch

    def combine_patches(*patches: Optional[Callable[[ExperimentConfig], None]]) -> Callable[[ExperimentConfig], None]:
        def _patch(cfg: ExperimentConfig) -> None:
            for fn in patches:
                if fn is not None:
                    fn(cfg)
        return _patch

    def seed_patch(seed: int) -> Callable[[ExperimentConfig], None]:
        def _patch(cfg: ExperimentConfig) -> None:
            cfg.random_seed = int(seed)
            cfg.data.random_seed = int(seed)
        return _patch

    def patch_shared_predictor(cfg: ExperimentConfig) -> None:
        cfg.model.encoder_config.setdefault('y_fusion', {})['enable_cross_predictor'] = False

    def patch_no_cv(cfg: ExperimentConfig) -> None:
        cfg.training.check_val_every_n_epoch = max(1, int(getattr(cfg.training, 'check_val_every_n_epoch', 1)))
        cfg.training.early_stopping_patience = 0
        cfg.training.save_every_n_epochs = max(1, int(getattr(cfg.training, 'save_every_n_epochs', 1)))

    def patch_k(cfg: ExperimentConfig, value: int) -> None:
        cfg.model.loss_config['mac']['max_pairs'] = value

    def patch_z(cfg: ExperimentConfig, value: int) -> None:
        cfg.model.z_dim = value

    table_specs = {
        'table1': {
            'headers': ['数据规模', 'CSI', 'MBRI', 'MAC', 'Structural', 'RIC-avg'],
            'rows': [
                ('n=500', spec_with_dataset('table1_n500', 'carl_full', 500, 0.3, 'quadratic')),
                ('n=1000', spec_with_dataset('table1_n1000', 'carl_full', 1000, 0.3, 'quadratic')),
                ('n=2000', spec_with_dataset('table1_n2000', 'carl_full', 2000, 0.3, 'quadratic')),
                ('n=5000', spec_with_dataset('table1_n5000', 'carl_full', 5000, 0.3, 'quadratic')),
            ]
        },
        'table2': {
            'headers': ['噪声水平', 'CSI', 'MBRI', 'MAC', 'Structural', 'RIC-avg'],
            'rows': [
                ('σ=0.1', spec_with_dataset('table2_sigma01', 'carl_full', 2000, 0.1, 'quadratic')),
                ('σ=0.3', spec_with_dataset('table2_sigma03', 'carl_full', 2000, 0.3, 'quadratic')),
                ('σ=0.5', spec_with_dataset('table2_sigma05', 'carl_full', 2000, 0.5, 'quadratic')),
            ]
        },
        'table3': {
            'headers': ['方法', 'CSI', 'MBRI', 'MAC', 'Structural', 'RIC-avg'],
            'rows': [
                ('CARL', spec_with_dataset('table3_carl', 'carl_full', 2000, 0.3, 'quadratic')),
                ('ALIGN', spec_with_dataset('table3_align', 'align_baseline', 2000, 0.3, 'quadratic')),
                ('CLIP', spec_with_dataset('table3_clip', 'clip_baseline', 2000, 0.3, 'quadratic')),
                ('ImageBind', spec_with_dataset('table3_imagebind', 'imagebind_baseline', 2000, 0.3, 'quadratic')),
                ('DCCA', spec_with_dataset('table3_dcca', 'dcca_baseline', 2000, 0.3, 'quadratic')),
                ('IRM', spec_with_dataset('table3_irm', 'irm_baseline', 2000, 0.3, 'quadratic')),
                ('CausalVAE', spec_with_dataset('table3_causalvae', 'causal_vae_baseline', 2000, 0.3, 'quadratic')),
                ('DEAR', spec_with_dataset('table3_dear', 'dear_baseline', 2000, 0.3, 'quadratic')),
                ('AutoEncoder', spec_with_dataset('table3_autoencoder', 'autoencoder_baseline', 2000, 0.3, 'quadratic')),
                ('Concat', spec_with_dataset('table3_concat', 'concat_baseline', 2000, 0.3, 'quadratic')),
            ]
        },
        'table5': {
            'headers': ['模型变体', 'CSI', 'MBRI', 'MAC', 'Structural', 'RIC-avg'],
            'rows': [
                ('CARL (Full)', spec_with_dataset('table5_full', 'carl_full', 2000, 0.3, 'quadratic')),
                ('w/o LCI', spec_with_dataset('table5_no_lci', 'carl_full', 2000, 0.3, 'quadratic', patch=loss_toggle_patch(disabled=['ci']))),
                ('w/o LMBR', spec_with_dataset('table5_no_mbr', 'carl_full', 2000, 0.3, 'quadratic', patch=loss_toggle_patch(disabled=['mbr']))),
                ('w/o LMAC', spec_with_dataset('table5_no_mac', 'carl_full', 2000, 0.3, 'quadratic', patch=loss_toggle_patch(disabled=['mac']))),
                ('Only Lalign', spec_with_dataset('table5_align_only', 'carl_full', 2000, 0.3, 'quadratic', patch=loss_toggle_patch(enabled_only=['align']))),
            ]
        },
        'table6': {
            'headers': ['设计选择', 'CSI', 'MBRI', 'MAC', 'Structural', 'RIC-avg'],
            'rows': [
                ('CARL (默认)', spec_with_dataset('table6_full', 'carl_full', 2000, 0.3, 'quadratic')),
                ('共享预测头', spec_with_dataset('table6_shared_head', 'carl_full', 2000, 0.3, 'quadratic', patch=patch_shared_predictor)),
                ('无交叉验证', spec_with_dataset('table6_no_cv', 'carl_full', 2000, 0.3, 'quadratic', patch=patch_no_cv)),
                ('K=32 (vs 128)', spec_with_dataset('table6_k32', 'carl_full', 2000, 0.3, 'quadratic', patch=lambda cfg: patch_k(cfg, 32))),
                ('d=16 (vs 64)', spec_with_dataset('table6_z16', 'carl_full', 2000, 0.3, 'quadratic', patch=lambda cfg: patch_z(cfg, 16))),
            ]
        },
        'table5_style_ib': {
            'headers': ['模型变体', 'CSI', 'MBRI', 'MAC', 'Structural', 'RIC-avg'],
            'rows': [
                ('基线(无Style/IB)', spec_with_dataset(
                    'table5_styleib_base',
                    'carl_full',
                    2000,
                    0.3,
                    'quadratic',
                    patch=compact_phase_patch()
                )),
                ('+ Style (风格一致性)', spec_with_dataset(
                    'table5_styleib_style_only',
                    'carl_full',
                    2000,
                    0.3,
                    'quadratic',
                    patch=combine_patches(
                        compact_phase_patch(),
                        enable_losses_patch(['style'], weight_overrides={'style': 0.1})
                    )
                )),
                ('+ IB (信息瓶颈)', spec_with_dataset(
                    'table5_styleib_ib_only',
                    'carl_full',
                    2000,
                    0.3,
                    'quadratic',
                    patch=combine_patches(
                        compact_phase_patch(),
                        enable_losses_patch(['ib'], weight_overrides={'ib': 0.1})
                    )
                )),
                ('+ Style + IB (全量)', spec_with_dataset(
                    'table5_styleib_full',
                    'carl_full',
                    2000,
                    0.3,
                    'quadratic',
                    patch=combine_patches(
                        compact_phase_patch(),
                        enable_losses_patch(['style', 'ib'], weight_overrides={'style': 0.1, 'ib': 0.1})
                    )
                )),
            ]
        }
    }

    for table_name, spec in table_specs.items():
        if tables_set and table_name not in tables_set:
            continue

        rows = []
        for label, exp_spec in spec['rows']:
            seed_metrics: List[Dict[str, float]] = []

            for seed in seeds_list:
                # Build a seed-specific spec (name + seed patch)
                seed_suffix = f"_seed{seed}" if seed is not None else ""
                seed_spec = ExperimentSpec(
                    name=f"{exp_spec.name}{seed_suffix}",
                    model_id=exp_spec.model_id,
                    overrides=exp_spec.overrides,
                    epochs=exp_spec.epochs,
                    data_dir=exp_spec.data_dir,
                    patch=combine_patches(exp_spec.patch, seed_patch(seed) if seed is not None else None),
                    seed=seed
                )

                output_dir = OUTPUT_ROOT / seed_spec.name
                history_path = output_dir / 'training_history.json'

                if skip_existing and history_path.exists():
                    metrics = extract_metrics(output_dir)
                else:
                    cfg_path = prepare_config(base_cfg, seed_spec)
                    run_training(cfg_path, seed_spec)
                    metrics = extract_metrics(output_dir)

                seed_metrics.append(metrics)

            def _aggregate(metric_key: str) -> str:
                vals = [m[metric_key] for m in seed_metrics if m[metric_key] == m[metric_key]]
                if not vals:
                    return ''
                if len(vals) == 1:
                    return f"{format_metric(vals[0]):.4f}"
                mean_val = sum(vals) / len(vals)
                min_val = min(vals)
                max_val = max(vals)
                return f"{mean_val:.4f} ({min_val:.4f}-{max_val:.4f})"

            row = [label] + [_aggregate(h) for h in spec['headers'][1:]]
            rows.append(row)

        result_name = f"{table_name}{'_seeded' if seeds else ''}.csv"
        write_table(RESULT_ROOT / result_name, spec['headers'], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CARL benchmark tables and export CSV files.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Epochs per experiment (default: %(default)s)')
    parser.add_argument('--tables', nargs='+', help='Optional subset of table names to run (e.g., table5_style_ib)')
    parser.add_argument('--seeds', nargs='+', type=int, help='Optional list of random seeds for multi-seed aggregation')
    parser.add_argument('--skip-existing', action='store_true', help='Reuse metrics if training_history.json already exists')
    args = parser.parse_args()
    run_tables(args.epochs, tables=args.tables, seeds=args.seeds, skip_existing=args.skip_existing)


if __name__ == '__main__':
    main()
