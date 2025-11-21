#!/usr/bin/env python3
"""Generate CSP datasets for CARL experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CONFIG_PATH = Path('csp_synth/configs/mnist_dual.yaml')
OUTPUT_ROOT = Path('csp_synth/generated')
SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
    n_samples: int
    noise: float
    nonlinearity: str = 'quadratic'

    def slug(self) -> str:
        sigma_str = f"{self.noise:.2f}".rstrip('0').rstrip('.')
        sigma_slug = sigma_str.replace('.', 'p')
        return f"n{self.n_samples}_sigma{sigma_slug}_{self.nonlinearity}"

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.slug()


def nonlinearity_overrides(spec: DatasetSpec) -> List[Tuple[str, str]]:
    nl = spec.nonlinearity.lower()
    if nl == 'quadratic':
        return [
            ('scm.h1', 'square'),
            ('scm.h2', 'square'),
            ('scm.gamma1', '1.0'),
            ('scm.gamma2', '1.0'),
        ]
    if nl == 'linear':
        return [
            ('scm.h1', 'tanh'),
            ('scm.h2', 'tanh'),
            ('scm.gamma1', '0.1'),
            ('scm.gamma2', '0.1'),
        ]
    if nl == 'neural':
        return [
            ('scm.h1', 'sin'),
            ('scm.h2', 'tanh'),
            ('scm.gamma1', '1.0'),
            ('scm.gamma2', '1.0'),
        ]
    raise ValueError(f"Unsupported nonlinearity: {spec.nonlinearity}")


def build_command(spec: DatasetSpec, force: bool, no_sanity: bool) -> List[str]:
    cmd: List[str] = [
        'python',
        'csp_synth/scripts/gen_dataset.py',
        '--config', str(CONFIG_PATH),
        '--seed-override', str(SEED),
        '--n-override', str(spec.n_samples),
        '--output-override', str(spec.output_dir),
        '--set', f'scm.sigma_T={spec.noise}',
        '--set', f'scm.sigma_M={spec.noise}',
        '--set', f'scm.sigma_Y={spec.noise}',
        '--set', f'imaging.sigma_pix={spec.noise / 10.0}',
    ]

    for key, value in nonlinearity_overrides(spec):
        cmd.extend(['--set', f'{key}={value}'])

    if force:
        cmd.append('--force')
    if no_sanity:
        cmd.append('--no-sanity')
    return cmd


def generate_dataset(spec: DatasetSpec, force: bool, no_sanity: bool) -> None:
    output_dir = spec.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and not force:
        print(f"Skipping existing dataset: {output_dir}")
        return

    cmd = build_command(spec, force=force, no_sanity=no_sanity)
    print(f"Generating dataset: {spec.slug()}\n  Dir: {output_dir}\n  Cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def save_manifest(specs: Iterable[DatasetSpec]) -> None:
    manifest_path = OUTPUT_ROOT / 'manifest.json'
    data: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        data[spec.slug()] = {
            'path': str(spec.output_dir.resolve()),
            'n_samples': spec.n_samples,
            'noise': spec.noise,
            'nonlinearity': spec.nonlinearity,
        }
    manifest_path.write_text(json.dumps(data, indent=2))
    print(f"Manifest saved to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate CARL experiment datasets.')
    parser.add_argument('--force', action='store_true', help='Overwrite existing datasets')
    parser.add_argument('--no-sanity', action='store_true', help='Skip dataset generation sanity checks')
    parser.add_argument('--include-linear', action='store_true', help='Also generate linear nonlinearity variants for default n/noise')
    parser.add_argument('--include-neural', action='store_true', help='Also generate neural nonlinearity variants for default n/noise')
    parser.add_argument('--table4-matrix', action='store_true', help='Generate full Table 4 dataset matrix (n ∈ {500,1000,2000,5000}, σ ∈ {0.1,0.3,0.5}, nl ∈ {linear,quadratic,neural})')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    required_specs = [
        DatasetSpec(500, 0.3, 'quadratic'),
        DatasetSpec(1000, 0.3, 'quadratic'),
        DatasetSpec(2000, 0.3, 'quadratic'),
        DatasetSpec(5000, 0.3, 'quadratic'),
        DatasetSpec(2000, 0.1, 'quadratic'),
        DatasetSpec(2000, 0.5, 'quadratic'),
    ]

    if args.include_linear:
        required_specs.extend([
            DatasetSpec(2000, 0.3, 'linear'),
        ])

    if args.include_neural:
        required_specs.extend([
            DatasetSpec(2000, 0.3, 'neural'),
        ])

    if args.table4_matrix:
        table4_specs = [
            DatasetSpec(n, sigma, nonlin)
            for n in [500, 1000, 2000, 5000]
            for sigma in [0.1, 0.3, 0.5]
            for nonlin in ['linear', 'quadratic', 'neural']
        ]
        required_specs.extend(table4_specs)

    # Remove duplicates while preserving order
    seen = set()
    unique_specs = []
    for spec in required_specs:
        slug = spec.slug()
        if slug in seen:
            continue
        seen.add(slug)
        unique_specs.append(spec)

    for spec in unique_specs:
        generate_dataset(spec, force=args.force, no_sanity=args.no_sanity)

    save_manifest(unique_specs)


if __name__ == '__main__':
    main()
