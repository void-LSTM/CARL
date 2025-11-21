"""Baseline model implementations for CSP experiments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import CSPEncoderModule, MLP


def _build_projection_head(z_dim: int, proj_dim: int, activation: str = 'relu', use_batch_norm: bool = True, dropout: float = 0.0) -> nn.Module:
    if proj_dim == z_dim and dropout == 0.0:
        # Identity mapping when dimensions match and no dropout requested
        return nn.Identity()

    hidden = max(proj_dim, z_dim)
    return MLP(
        input_dim=z_dim,
        output_dim=proj_dim,
        hidden_dims=[hidden],
        activation=activation,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        final_activation=False,
    )


class ConcatBaselineModel(nn.Module):
    """Simple baseline that concatenates modality encodings and predicts targets."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        predictor_config: Optional[Dict[str, Any]] = None,
        extra_outputs: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.z_dim = z_dim

        encoder_cfg = self._build_encoder_config(encoder_config)
        self.encoders = CSPEncoderModule(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_configs=encoder_cfg,
        )
        self.feature_dims = feature_dims

        baseline_cfg = predictor_config or {}
        hidden_dims = baseline_cfg.get('hidden_dims', [128, 64])
        activation = baseline_cfg.get('activation', 'relu')
        dropout = float(baseline_cfg.get('dropout', 0.1))
        use_batch_norm = bool(baseline_cfg.get('use_batch_norm', True))
        output_dim = feature_dims.get('Y_dim', 1) or 1

        concat_sources = baseline_cfg.get('concat_sources')
        if concat_sources is None:
            concat_sources = ['z_T', 'z_M', 'z_Y']
        self.concat_sources = self._normalize_concat_sources(list(concat_sources))

        concat_input_dim = 0
        for key in self.concat_sources:
            concat_input_dim += z_dim
        if concat_input_dim == 0:
            raise ValueError("ConcatBaselineModel requires at least one source representation")

        self.predictor = MLP(
            input_dim=concat_input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            final_activation=False,
        )

        self.training_phases = {
            'full': {
                'enabled_losses': ['recon'],
                'epochs': [0, float('inf')],
            }
        }

        self.extra_outputs = list(extra_outputs) if extra_outputs is not None else []
        self.current_step = 0

    @staticmethod
    def _default_encoder_config() -> Dict[str, Any]:
        return {
            'tabular': {
                'hidden_dims': [128, 128],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout': 0.1,
            },
            'image': {
                'architecture': 'small_cnn',
                'use_batch_norm': True,
                'dropout': 0.1,
            },
            'y_fusion': {
                'enabled': True,
                'force_mediator': False,
                'direct_scale': 0.5,
                'direct_dropout': 0.2,
                'gate_sharpness': 2.0,
                'mediator_scale': 1.0,
            },
        }

    @classmethod
    def _merge_dict(cls, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = cls._merge_dict(base[key], value)
            else:
                base[key] = value
        return base

    @classmethod
    def _build_encoder_config(cls, encoder_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base_cfg = cls._default_encoder_config()
        if not encoder_config:
            return base_cfg

        cfg = deepcopy(base_cfg)

        # Accept both hierarchical and flattened overrides
        flattened_keys = {'hidden_dims', 'activation', 'use_batch_norm', 'dropout'}
        if any(key in encoder_config for key in flattened_keys):
            tab_cfg = {k: v for k, v in encoder_config.items() if k in flattened_keys}
            cfg['tabular'] = cls._merge_dict(cfg['tabular'], tab_cfg)
            cfg['image'] = cls._merge_dict(
                cfg['image'],
                {k: v for k, v in tab_cfg.items() if k in {'activation', 'use_batch_norm', 'dropout'}},
            )

        y_fusion_cfg = encoder_config.get('y_fusion') if isinstance(encoder_config, dict) else None
        if isinstance(y_fusion_cfg, dict):
            cfg['y_fusion'] = cls._merge_dict(cfg['y_fusion'], y_fusion_cfg)

        hierarchical_keys = {k: v for k, v in encoder_config.items() if isinstance(v, dict) and k in {'tabular', 'image'}}
        if hierarchical_keys:
            cfg = cls._merge_dict(cfg, hierarchical_keys)

        return cfg

    def _gather_concat_sources(self, representations: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self.concat_sources:
            if key not in representations:
                raise KeyError(f"Expected representation '{key}' in encoder output")
            parts.append(representations[key])
        return torch.cat(parts, dim=1)

    @staticmethod
    def _normalize_concat_sources(sources: Iterable[str]) -> List[str]:
        """Expand helper aliases and remove duplicates while preserving order."""
        expanded: List[str] = []
        for key in sources:
            if key == 'z_concat':
                for alias in ('z_T', 'z_M', 'z_Y'):
                    if alias not in expanded:
                        expanded.append(alias)
                continue
            if key not in expanded:
                expanded.append(key)
        return expanded

    def update_step(self, step: int) -> None:
        self.current_step = step


class ContrastiveBaselineModel(nn.Module):
    """Generic contrastive baseline using InfoNCE across representation pairs."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        contrastive_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.z_dim = z_dim
        contrastive_config = contrastive_config or {}

        encoder_cfg = ConcatBaselineModel._build_encoder_config(encoder_config)
        self.encoders = CSPEncoderModule(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_configs=encoder_cfg,
        )
        self.feature_dims = feature_dims

        self.pairs = [tuple(pair) for pair in contrastive_config.get('pairs', [('z_M', 'z_Y')])]
        self.temperature = float(contrastive_config.get('temperature', 0.07))
        self.projection_dim = int(contrastive_config.get('projection_dim', z_dim))
        self.normalize = bool(contrastive_config.get('normalize', True))
        self.loss_weight = float(contrastive_config.get('loss_weight', 1.0))
        proj_activation = contrastive_config.get('projection_activation', 'relu')
        proj_dropout = float(contrastive_config.get('projection_dropout', 0.0))
        proj_batch_norm = bool(contrastive_config.get('projection_use_batch_norm', True))

        unique_keys = sorted({rep for pair in self.pairs for rep in pair})
        self.projectors = nn.ModuleDict()
        for key in unique_keys:
            self.projectors[key] = _build_projection_head(
                z_dim,
                self.projection_dim,
                activation=proj_activation,
                use_batch_norm=proj_batch_norm,
                dropout=proj_dropout,
            )

        self.current_step = 0
        self.eps = 1e-8

    def update_step(self, step: int) -> None:
        self.current_step = step

    def _get_representation(self, representations: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
        if key in representations:
            return representations[key]
        # Support accessing image-specific keys with fallbacks
        fallback_map = {
            'z_I_M': representations.get('z_M') if 'z_M' in representations else None,
            'z_I_Y': representations.get('z_Y') if 'z_Y' in representations else None,
        }
        return fallback_map.get(key)

    def _info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            z1 = F.normalize(z1, dim=1, eps=self.eps)
            z2 = F.normalize(z2, dim=1, eps=self.eps)

        logits = z1 @ z2.T / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss12 = F.cross_entropy(logits, labels)
        loss21 = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss12 + loss21)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        representations = self.encoders(batch)
        pair_losses = []
        pair_metrics = {}

        for a_key, b_key in self.pairs:
            z_a = self._get_representation(representations, a_key)
            z_b = self._get_representation(representations, b_key)
            if z_a is None or z_b is None:
                continue

            proj_a = self.projectors[a_key](z_a)
            proj_b = self.projectors[b_key](z_b)
            loss = self._info_nce_loss(proj_a, proj_b)
            pair_losses.append(loss)
            pair_metrics[f'{a_key}_vs_{b_key}_loss'] = loss.detach()

        if pair_losses:
            contrastive_loss = torch.stack(pair_losses).mean() * self.loss_weight
        else:
            contrastive_loss = torch.zeros((), device=next(self.parameters()).device)

        outputs: Dict[str, torch.Tensor] = {
            **representations,
            'contrastive_loss': contrastive_loss,
            'total_loss': contrastive_loss,
        }
        for key, value in pair_metrics.items():
            outputs[key] = value

        return outputs


class AlignBaselineModel(ContrastiveBaselineModel):
    """ALIGN-style baseline aligning treatment representations with mediator/outcome encodings."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        align_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        default_config = {
            'pairs': [('z_T', 'z_M'), ('z_T', 'z_Y')],
            'temperature': 0.07,
            'projection_dim': z_dim,
            'loss_weight': 1.0,
        }
        if align_config:
            default_config.update(align_config)
        super().__init__(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_config=encoder_config,
            contrastive_config=default_config,
        )


class ClipBaselineModel(ContrastiveBaselineModel):
    """CLIP-style contrastive baseline aligning mediator and outcome representations."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        clip_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        default_config = {
            'pairs': [('z_M', 'z_Y')],
            'temperature': 0.07,
            'projection_dim': z_dim,
            'loss_weight': 1.0,
        }
        if clip_config:
            default_config.update(clip_config)
        super().__init__(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_config=encoder_config,
            contrastive_config=default_config,
        )


class ImageBindBaselineModel(ContrastiveBaselineModel):
    """ImageBind-style baseline leveraging image encoders alongside mediator/outcome signals."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        imagebind_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        default_pairs = [('z_I_M', 'z_M'), ('z_I_M', 'z_Y')]
        if scenario in ['IY', 'DUAL']:
            default_pairs.append(('z_I_Y', 'z_Y'))
        default_config = {
            'pairs': default_pairs,
            'temperature': 0.05,
            'projection_dim': z_dim,
            'loss_weight': 1.0,
        }
        if imagebind_config:
            default_config.update(imagebind_config)
        super().__init__(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_config=encoder_config,
            contrastive_config=default_config,
        )


class AutoEncoderBaselineModel(nn.Module):
    """Standard autoencoder baseline with shared encoders and MLP decoders."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        ae_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.z_dim = z_dim

        encoder_cfg = ConcatBaselineModel._build_encoder_config(encoder_config)
        self.encoders = CSPEncoderModule(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_configs=encoder_cfg,
        )
        self.feature_dims = feature_dims
        self.current_step = 0

        ae_cfg = deepcopy(ae_config or {})
        self.concat_sources = list(ae_cfg.get('concat_sources', ['z_T', 'z_M', 'z_Y']))
        if not self.concat_sources:
            raise ValueError("AutoEncoderBaselineModel requires concat_sources to be non-empty")

        fusion_hidden = ae_cfg.get('fusion_hidden_dims', [256, 128])
        fusion_activation = ae_cfg.get('fusion_activation', 'relu')
        fusion_dropout = float(ae_cfg.get('fusion_dropout', 0.1))
        fusion_batch_norm = bool(ae_cfg.get('fusion_use_batch_norm', True))

        concat_input_dim = len(self.concat_sources) * z_dim
        self.latent_encoder = MLP(
            input_dim=concat_input_dim,
            output_dim=z_dim,
            hidden_dims=fusion_hidden,
            activation=fusion_activation,
            use_batch_norm=fusion_batch_norm,
            dropout=fusion_dropout,
            final_activation=False,
        )

        decoder_hidden = ae_cfg.get('decoder_hidden_dims', [128, 64])
        decoder_activation = ae_cfg.get('decoder_activation', 'relu')
        decoder_dropout = float(ae_cfg.get('decoder_dropout', 0.1))
        decoder_batch_norm = bool(ae_cfg.get('decoder_use_batch_norm', True))

        default_targets = ['Y_star']
        decoder_targets = list(ae_cfg.get('decoder_targets', default_targets))
        if not decoder_targets:
            decoder_targets = default_targets
        self.decoder_targets = decoder_targets

        self.decoders = nn.ModuleDict()
        for target in self.decoder_targets:
            out_dim = self._infer_output_dim(target, feature_dims)
            self.decoders[target] = MLP(
                input_dim=z_dim,
                output_dim=out_dim,
                hidden_dims=decoder_hidden,
                activation=decoder_activation,
                use_batch_norm=decoder_batch_norm,
                dropout=decoder_dropout,
                final_activation=False,
            )

        ae_loss_weights = ae_cfg.get('loss_weights', {})
        self.loss_weights = {target: float(ae_loss_weights.get(target, 1.0)) for target in self.decoder_targets}

    @staticmethod
    def _infer_output_dim(target: str, feature_dims: Dict[str, int]) -> int:
        if target == 'Y_star':
            return feature_dims.get('Y_dim', 1) or 1
        if target == 'M':
            return feature_dims.get('M_dim', 1) or 1
        if target == 'T':
            return feature_dims.get('T_dim', 1) or 1
        raise ValueError(f"Unsupported decoder target '{target}' for AutoEncoder baseline")

    def _gather_concat_sources(self, representations: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self.concat_sources:
            if key not in representations:
                raise KeyError(f"Expected representation '{key}' in encoder output")
            parts.append(representations[key])
        return torch.cat(parts, dim=1)

    def update_step(self, step: int) -> None:
        self.current_step = step

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        representations = self.encoders(batch)

        z_concat = self._gather_concat_sources(representations)
        z_latent = self.latent_encoder(z_concat)

        outputs: Dict[str, torch.Tensor] = {**representations, 'z_concat': z_concat, 'z_latent': z_latent}

        total_loss = 0.0
        recon_stats = {}
        for target in self.decoder_targets:
            decoder = self.decoders[target]
            prediction = decoder(z_latent)
            outputs[f'{target}_recon'] = prediction

            target_tensor = batch.get(target)
            if target_tensor is None:
                raise KeyError(f"Batch missing '{target}' key required for reconstruction")

            pred_flat = prediction.view(prediction.size(0), -1)
            target_flat = target_tensor.view(target_tensor.size(0), -1).float()

            mse = F.mse_loss(pred_flat, target_flat)
            weight = self.loss_weights.get(target, 1.0)
            total_loss = total_loss + weight * mse
            outputs[f'{target}_mse'] = mse
            recon_stats[target] = {
                'mse': mse,
                'mae': torch.mean(torch.abs(pred_flat - target_flat))
            }
            outputs[f'{target}_mae'] = recon_stats[target]['mae']

        outputs['total_loss'] = total_loss
        return outputs

class DCCABaselineModel(nn.Module):
    """Deep CCA style baseline maximizing correlation between mediator and outcome reps."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        cca_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.z_dim = z_dim

        encoder_cfg = ConcatBaselineModel._build_encoder_config(encoder_config)
        self.encoders = CSPEncoderModule(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_configs=encoder_cfg,
        )
        self.feature_dims = feature_dims
        self.current_step = 0

        cfg = cca_config or {}
        self.outdim = int(cfg.get('outdim', min(10, z_dim)))
        self.reg1 = float(cfg.get('reg1', 1e-3))
        self.reg2 = float(cfg.get('reg2', 1e-3))
        self.eps = float(cfg.get('eps', 1e-6))
        self.detach_targets = bool(cfg.get('detach_targets', False))
        self.normalize = bool(cfg.get('normalize', True))

    def update_step(self, step: int) -> None:
        self.current_step = step

    def _center(self, h: torch.Tensor) -> torch.Tensor:
        return h - h.mean(dim=0, keepdim=True)

    def _compute_correlation(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            h1 = F.normalize(h1, dim=1, eps=self.eps)
            h2 = F.normalize(h2, dim=1, eps=self.eps)

        h1c = self._center(h1)
        h2c = self._center(h2)

        std1 = torch.sqrt(torch.clamp((h1c ** 2).mean(dim=0), min=self.eps))
        std2 = torch.sqrt(torch.clamp((h2c ** 2).mean(dim=0), min=self.eps))

        corr_vec = (h1c * h2c).mean(dim=0) / (std1 * std2)
        corr_vec = torch.nan_to_num(corr_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return corr_vec

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        representations = self.encoders(batch)

        h_m = representations.get('z_M')
        h_y = representations.get('z_Y')
        if h_m is None or h_y is None:
            raise KeyError("DCCABaselineModel requires 'z_M' and 'z_Y' representations from encoders")

        corr_vec = self._compute_correlation(h_m, h_y)
        topk = torch.topk(corr_vec.abs(), k=min(self.outdim, corr_vec.numel())).values
        corr_score = topk.mean()
        loss = 1.0 - corr_score

        outputs: Dict[str, torch.Tensor] = {
            **representations,
            'cca_correlation': corr_vec,
            'cca_loss': loss,
            'total_loss': loss,
        }

        return outputs


class IRMBaselineModel(nn.Module):
    """Invariant Risk Minimization baseline with simple environment splitting on treatment."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        irm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.z_dim = z_dim

        irm_cfg = deepcopy(irm_config or {})
        self.target_key = irm_cfg.get('target_key', 'Y_star')
        self.environment_key = irm_cfg.get('environment_key', 'T')
        self.loss_type = irm_cfg.get('loss_type', 'mse')
        self.irm_lambda = float(irm_cfg.get('irm_lambda', 1.0))
        self.dropout = float(irm_cfg.get('dropout', 0.1))
        predictor_hidden = irm_cfg.get('hidden_dims', [128, 64])
        activation = irm_cfg.get('activation', 'relu')
        use_batch_norm = bool(irm_cfg.get('use_batch_norm', True))

        encoder_cfg = ConcatBaselineModel._build_encoder_config(encoder_config)
        self.encoders = CSPEncoderModule(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_configs=encoder_cfg,
        )
        self.feature_dims = feature_dims

        output_dim = feature_dims.get('Y_dim', 1) or 1
        self.predictor = MLP(
            input_dim=z_dim,
            output_dim=output_dim,
            hidden_dims=predictor_hidden,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=self.dropout,
            final_activation=False,
        )

    def update_step(self, step: int) -> None:
        # No internal state to update, method provided for trainer compatibility
        return None

    def _environment_masks(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        env_tensor = batch.get(self.environment_key)
        if env_tensor is None:
            return []
        env_tensor = env_tensor.view(-1)
        masks = []
        masks.append(env_tensor <= env_tensor.median())
        masks.append(env_tensor > env_tensor.median())
        return masks

    def _loss_fn(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mae':
            return F.l1_loss(prediction, target)
        return F.mse_loss(prediction, target)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        representations = self.encoders(batch)
        z_y = representations.get('z_Y')
        if z_y is None:
            raise KeyError("IRMBaselineModel requires 'z_Y' representation")

        prediction = self.predictor(z_y)
        target = batch.get(self.target_key)
        if target is None:
            raise KeyError(f"IRMBaselineModel missing target '{self.target_key}' in batch")
        target = target.view_as(prediction).float()

        env_masks = self._environment_masks(batch)
        losses = []
        penalty = torch.zeros((), device=prediction.device)

        if self.training:
            for mask in env_masks:
                if mask is None or mask.sum() == 0:
                    continue
                mask = mask.to(prediction.device)
                env_pred = prediction[mask]
                env_target = target[mask]
                scale = torch.tensor(1.0, device=prediction.device, requires_grad=True)
                env_loss = self._loss_fn(env_pred * scale, env_target)
                losses.append(env_loss)
                grad = torch.autograd.grad(env_loss, scale, create_graph=True)[0]
                penalty = penalty + grad.pow(2)
        else:
            for mask in env_masks:
                if mask is None or mask.sum() == 0:
                    continue
                mask = mask.to(prediction.device)
                env_pred = prediction[mask]
                env_target = target[mask]
                losses.append(self._loss_fn(env_pred, env_target))

        if losses:
            empirical_risk = torch.stack(losses).mean()
        else:
            empirical_risk = self._loss_fn(prediction, target)

        total_loss = empirical_risk + (self.irm_lambda * penalty if self.training else 0.0)

        outputs: Dict[str, torch.Tensor] = {
            **representations,
            'irm_prediction': prediction,
            'irm_penalty': penalty.detach() if torch.is_tensor(penalty) else torch.tensor(0.0),
            'irm_empirical_risk': empirical_risk.detach(),
            'total_loss': total_loss,
        }

        return outputs


class CausalVAEBaselineModel(nn.Module):
    """Simplified CausalVAE-inspired baseline modelling mediator-outcome generative process."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        vae_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.z_dim = z_dim

        encoder_cfg = ConcatBaselineModel._build_encoder_config(encoder_config)
        self.encoders = CSPEncoderModule(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_configs=encoder_cfg,
        )
        self.feature_dims = feature_dims
        self.current_step = 0

        cfg = deepcopy(vae_config or {})
        self.latent_dim = int(cfg.get('latent_dim', 32))
        self.beta = float(cfg.get('beta', 1.0))
        self.concat_sources = list(cfg.get('concat_sources', ['z_T', 'z_M']))
        if 'z_Y' not in self.concat_sources:
            self.concat_sources.append('z_Y')

        decoder_hidden = cfg.get('decoder_hidden_dims', [128, 64])
        decoder_activation = cfg.get('decoder_activation', 'relu')
        decoder_dropout = float(cfg.get('decoder_dropout', 0.1))
        decoder_batch_norm = bool(cfg.get('decoder_use_batch_norm', True))

        concat_dim = len(self.concat_sources) * z_dim
        self.encoder_mu = nn.Linear(concat_dim, self.latent_dim)
        self.encoder_logvar = nn.Linear(concat_dim, self.latent_dim)
        self.decoder = MLP(
            input_dim=self.latent_dim,
            output_dim=feature_dims.get('Y_dim', 1) or 1,
            hidden_dims=decoder_hidden,
            activation=decoder_activation,
            use_batch_norm=decoder_batch_norm,
            dropout=decoder_dropout,
            final_activation=False,
        )

    def update_step(self, step: int) -> None:
        self.current_step = step

    def _gather_concat(self, representations: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self.concat_sources:
            if key not in representations:
                raise KeyError(f"CausalVAEBaselineModel missing representation '{key}'")
            parts.append(representations[key])
        return torch.cat(parts, dim=1)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        representations = self.encoders(batch)
        concat = self._gather_concat(representations)

        mu = self.encoder_mu(concat)
        logvar = self.encoder_logvar(concat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std

        recon = self.decoder(latent)
        target = batch.get('Y_star')
        if target is None:
            raise KeyError("CausalVAEBaselineModel requires 'Y_star' in batch")
        target = target.view_as(recon).float()

        recon_loss = F.mse_loss(recon, target)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl

        outputs: Dict[str, torch.Tensor] = {
            **representations,
            'vae_mu': mu,
            'vae_logvar': logvar,
            'vae_latent': latent,
            'vae_recon': recon,
            'vae_recon_loss': recon_loss.detach(),
            'vae_kl': kl.detach(),
            'total_loss': total_loss,
        }

        return outputs


class DEARBaselineModel(nn.Module):
    """Simplified DEAR baseline encouraging disentangled mediator and outcome representations."""

    def __init__(
        self,
        *,
        scenario: str,
        z_dim: int,
        feature_dims: Dict[str, int],
        encoder_config: Optional[Dict[str, Any]] = None,
        dear_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.z_dim = z_dim

        encoder_cfg = ConcatBaselineModel._build_encoder_config(encoder_config)
        self.encoders = CSPEncoderModule(
            scenario=scenario,
            z_dim=z_dim,
            feature_dims=feature_dims,
            encoder_configs=encoder_cfg,
        )
        self.feature_dims = feature_dims
        self.current_step = 0

        cfg = deepcopy(dear_config or {})
        predictor_hidden = cfg.get('hidden_dims', [128, 64])
        activation = cfg.get('activation', 'relu')
        dropout = float(cfg.get('dropout', 0.1))
        use_batch_norm = bool(cfg.get('use_batch_norm', True))
        self.disentangle_weight = float(cfg.get('disentangle_weight', 1.0))
        self.reconstruction_weight = float(cfg.get('reconstruction_weight', 1.0))
        self.target_key = cfg.get('target_key', 'Y_star')

        self.predictor = MLP(
            input_dim=z_dim,
            output_dim=feature_dims.get('Y_dim', 1) or 1,
            hidden_dims=predictor_hidden,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            final_activation=False,
        )

    def update_step(self, step: int) -> None:
        self.current_step = step

    def _disentangle_penalty(self, rep_a: torch.Tensor, rep_b: torch.Tensor) -> torch.Tensor:
        rep_a = rep_a - rep_a.mean(dim=0, keepdim=True)
        rep_b = rep_b - rep_b.mean(dim=0, keepdim=True)
        cov = (rep_a.T @ rep_b) / (rep_a.size(0) - 1)
        return (cov.pow(2)).mean()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        representations = self.encoders(batch)
        z_m = representations.get('z_M')
        z_y = representations.get('z_Y')
        if z_m is None or z_y is None:
            raise KeyError("DEARBaselineModel requires 'z_M' and 'z_Y'")

        prediction = self.predictor(z_y)
        target = batch.get(self.target_key)
        if target is None:
            raise KeyError(f"DEARBaselineModel missing target '{self.target_key}'")
        target = target.view_as(prediction).float()

        recon_loss = F.mse_loss(prediction, target)
        disentangle_penalty = self._disentangle_penalty(z_m, z_y)
        total_loss = self.reconstruction_weight * recon_loss + self.disentangle_weight * disentangle_penalty

        outputs: Dict[str, torch.Tensor] = {
            **representations,
            'dear_prediction': prediction,
            'dear_recon_loss': recon_loss.detach(),
            'dear_disentangle': disentangle_penalty.detach(),
            'total_loss': total_loss,
        }

        return outputs


__all__ = [
    'ConcatBaselineModel',
    'AutoEncoderBaselineModel',
    'AlignBaselineModel',
    'ClipBaselineModel',
    'ImageBindBaselineModel',
    'DCCABaselineModel',
    'IRMBaselineModel',
    'CausalVAEBaselineModel',
    'DEARBaselineModel',
]
