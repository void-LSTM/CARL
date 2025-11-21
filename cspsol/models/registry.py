"""Model registry and factory utilities for CSP experiments.

Provides a central place to declare baseline models and CARL variants so
training/evaluation pipelines can instantiate models by identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
from copy import deepcopy

import torch.nn as nn

from .carl import CausalAwareModel
from .baselines import (
    ConcatBaselineModel,
    AutoEncoderBaselineModel,
    AlignBaselineModel,
    ClipBaselineModel,
    ImageBindBaselineModel,
    DCCABaselineModel,
    IRMBaselineModel,
    CausalVAEBaselineModel,
    DEARBaselineModel,
)


@dataclass
class ModelSpec:
    """Description of a model entry in the registry."""

    model_id: str
    build_fn: Callable[..., nn.Module]
    description: str = ''
    tags: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)

    def build(self,
              *,
              model_config: Any,
              feature_dims: Dict[str, Any],
              run_config: Optional[Any] = None,
              overrides: Optional[Dict[str, Any]] = None,
              device: Optional[Any] = None) -> nn.Module:
        """Instantiate the model using the stored builder."""
        params = deepcopy(self.default_params)
        if overrides:
            params.update(overrides)

        model = self.build_fn(
            model_config=model_config,
            feature_dims=feature_dims,
            run_config=run_config,
            **params,
        )

        if device is not None:
            model = model.to(device)
        return model


_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(model_id: str,
                   *,
                   description: str = '',
                   tags: Optional[Iterable[str]] = None,
                   default_params: Optional[Dict[str, Any]] = None) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a model builder in the registry."""

    def decorator(func: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if model_id in _MODEL_REGISTRY:
            raise ValueError(f"Model '{model_id}' is already registered")
        _MODEL_REGISTRY[model_id] = ModelSpec(
            model_id=model_id,
            build_fn=func,
            description=description,
            tags=list(tags) if tags else [],
            default_params=default_params or {},
        )
        return func

    return decorator


def get_model_spec(model_id: str) -> ModelSpec:
    """Return the registry specification for the requested model."""
    try:
        return _MODEL_REGISTRY[model_id]
    except KeyError as exc:
        available = ', '.join(sorted(_MODEL_REGISTRY.keys())) or '<empty>'
        raise KeyError(f"Unknown model_id '{model_id}'. Registered models: {available}") from exc


def list_registered_models() -> List[str]:
    """List all registered model identifiers."""
    return sorted(_MODEL_REGISTRY.keys())


def build_registered_model(model_id: str,
                           *,
                           model_config: Any,
                           feature_dims: Dict[str, Any],
                           run_config: Optional[Any] = None,
                           overrides: Optional[Dict[str, Any]] = None,
                           device: Optional[Any] = None) -> nn.Module:
    """Instantiate a model from the registry."""
    spec = get_model_spec(model_id)
    return spec.build(
        model_config=model_config,
        feature_dims=feature_dims,
        run_config=run_config,
        overrides=overrides,
        device=device,
    )


@register_model(
    'carl_full',
    description='CARL model with the full loss stack enabled',
    tags=['carl', 'baseline'],
)
def build_carl_full(*,
                    model_config: Any,
                    feature_dims: Dict[str, Any],
                    run_config: Optional[Any] = None) -> nn.Module:
    """Factory for the canonical CARL configuration."""
    loss_config = deepcopy(getattr(model_config, 'loss_config', {}))
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    balancer_config = deepcopy(getattr(model_config, 'balancer_config', {}))
    training_phases = deepcopy(getattr(model_config, 'phase_config', {}))

    # Allow run-level overrides for quick ablations
    if run_config is not None:
        variant = getattr(run_config, 'variant', None)
        if variant and variant != 'full':
            # Downstream variants will handle specific toggles in dedicated builders.
            # For safety we warn if someone tries to reuse this builder for ablations.
            import warnings

            warnings.warn(
                f"Variant '{variant}' requested but only 'full' is supported by 'carl_full' builder.",
                RuntimeWarning,
            )

    model = CausalAwareModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        loss_config=loss_config,
        encoder_config=encoder_config,
        balancer_config=balancer_config,
        training_phases=training_phases,
    )

    return model


@register_model(
    'concat_baseline',
    description='Concatenate modality encodings with an MLP predictor',
    tags=['baseline', 'simple'],
)
def build_concat_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    predictor_overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    predictor_config = deepcopy(baseline_config.get('predictor', {}))
    extra_outputs = baseline_config.get('extra_outputs')

    if predictor_overrides:
        predictor_config.update(predictor_overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            predictor_config.update(extra_params.get('predictor', {}))
            concat_sources = extra_params.get('concat_sources')
            if concat_sources is not None:
                predictor_config['concat_sources'] = concat_sources
            if 'extra_outputs' in extra_params and extra_outputs is None:
                extra_outputs = extra_params['extra_outputs']

    return ConcatBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        predictor_config=predictor_config,
        extra_outputs=extra_outputs,
    )


@register_model(
    'autoencoder_baseline',
    description='Shared-encoder autoencoder baseline without causal objectives',
    tags=['baseline', 'autoencoder'],
)
def build_autoencoder_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    ae_config = deepcopy(baseline_config.get('autoencoder', {}))

    if overrides:
        ae_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            ae_config.update(extra_params.get('autoencoder', {}))

    return AutoEncoderBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        ae_config=ae_config,
    )


@register_model(
    'align_baseline',
    description='ALIGN-style contrastive baseline',
    tags=['baseline', 'contrastive'],
)
def build_align_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    align_config = deepcopy(baseline_config.get('align', {}))

    if overrides:
        align_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            align_config.update(extra_params.get('align', {}))

    return AlignBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        align_config=align_config,
    )


@register_model(
    'clip_baseline',
    description='CLIP-style contrastive baseline',
    tags=['baseline', 'contrastive'],
)
def build_clip_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    clip_config = deepcopy(baseline_config.get('clip', {}))

    if overrides:
        clip_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            clip_config.update(extra_params.get('clip', {}))

    return ClipBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        clip_config=clip_config,
    )


@register_model(
    'imagebind_baseline',
    description='ImageBind-inspired multi-modal contrastive baseline',
    tags=['baseline', 'contrastive', 'multimodal'],
)
def build_imagebind_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    imagebind_config = deepcopy(baseline_config.get('imagebind', {}))

    if overrides:
        imagebind_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            imagebind_config.update(extra_params.get('imagebind', {}))

    return ImageBindBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        imagebind_config=imagebind_config,
    )


@register_model(
    'dcca_baseline',
    description='Deep CCA baseline aligning mediator and outcome representations',
    tags=['baseline', 'correlation'],
)
def build_dcca_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    cca_config = deepcopy(baseline_config.get('dcca', {}))

    if overrides:
        cca_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            cca_config.update(extra_params.get('dcca', {}))

    return DCCABaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        cca_config=cca_config,
    )


@register_model(
    'irm_baseline',
    description='Invariant Risk Minimization baseline',
    tags=['baseline', 'regularization'],
)
def build_irm_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    irm_config = deepcopy(baseline_config.get('irm', {}))

    if overrides:
        irm_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            irm_config.update(extra_params.get('irm', {}))

    return IRMBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        irm_config=irm_config,
    )


@register_model(
    'causal_vae_baseline',
    description='CausalVAE-style latent generative baseline',
    tags=['baseline', 'generative'],
)
def build_causal_vae_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    vae_config = deepcopy(baseline_config.get('causal_vae', {}))

    if overrides:
        vae_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            vae_config.update(extra_params.get('causal_vae', {}))

    return CausalVAEBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        vae_config=vae_config,
    )


@register_model(
    'dear_baseline',
    description='DEAR-style disentanglement baseline',
    tags=['baseline', 'disentanglement'],
)
def build_dear_baseline(
    *,
    model_config: Any,
    feature_dims: Dict[str, Any],
    run_config: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    encoder_config = deepcopy(getattr(model_config, 'encoder_config', {}))
    baseline_config = deepcopy(getattr(model_config, 'baseline_config', {}))
    dear_config = deepcopy(baseline_config.get('dear', {}))

    if overrides:
        dear_config.update(overrides)

    if run_config is not None:
        extra_params = getattr(run_config, 'extra_params', None)
        if isinstance(extra_params, dict):
            dear_config.update(extra_params.get('dear', {}))

    return DEARBaselineModel(
        scenario=getattr(model_config, 'scenario', 'IM'),
        z_dim=getattr(model_config, 'z_dim', 128),
        feature_dims=feature_dims,
        encoder_config=encoder_config,
        dear_config=dear_config,
    )


def ensure_default_models_registered() -> None:
    """Utility to ensure module side effects are triggered when imported."""
    # Function intentionally empty; importing this module registers defaults via decorators.
    return None
