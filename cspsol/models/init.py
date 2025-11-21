"""
Model implementations for CSP framework.
Core CARL model with encoders, losses, and training components.
"""

from .carl import CausalAwareModel
from .encoders import (
    CSPEncoderModule,
    TabularEncoder,
    ImageEncoder,
    FusionEncoder,
    MLP
)
from .losses import (
    LossCI,
    LossMBR, 
    LossMAC,
    LossAlign,
    LossStyle,
    LossIB,
    CMIEstimator,
    MIEstimator,
    GaussianHead,
    BernoulliHead
)
from .gradnorm import (
    MultiTaskBalancer,
    GradNorm,
    DynamicWeightAveraging
)
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
from .registry import (
    ModelSpec,
    register_model,
    list_registered_models,
    build_registered_model
)

__all__ = [
    # Main model
    'CausalAwareModel',
    
    # Encoders
    'CSPEncoderModule',
    'TabularEncoder',
    'ImageEncoder', 
    'FusionEncoder',
    'MLP',
    
    # Loss functions
    'LossCI',
    'LossMBR',
    'LossMAC', 
    'LossAlign',
    'LossStyle',
    'LossIB',
    'CMIEstimator',
    'MIEstimator',
    'GaussianHead',
    'BernoulliHead',

    # Multi-task balancing
    'MultiTaskBalancer',
    'GradNorm',
    'DynamicWeightAveraging',

    # Registry utilities
    'ModelSpec',
    'register_model',
    'list_registered_models',
    'build_registered_model',

    # Baselines
    'ConcatBaselineModel',
    'AutoEncoderBaselineModel',
    'AlignBaselineModel',
    'ClipBaselineModel',
    'ImageBindBaselineModel',
    'DCCABaselineModel',
    'IRMBaselineModel',
    'CausalVAEBaselineModel',
    'DEARBaselineModel'
]

# Model factory functions
def create_carl_model(scenario, z_dim=64, feature_dims=None, **kwargs):
    """
    Factory function to create CARL model with sensible defaults.
    
    Args:
        scenario: Training scenario ('IM', 'IY', 'DUAL')
        z_dim: Latent representation dimension
        feature_dims: Feature dimensions from data
        **kwargs: Additional model configuration
        
    Returns:
        CausalAwareModel instance
    """
    if feature_dims is None:
        feature_dims = {
            'T_dim': 1, 'M_dim': 1, 'Y_dim': 1,
            'img_channels': 1, 'img_height': 28, 'img_width': 28
        }
    
    return CausalAwareModel(
        scenario=scenario,
        z_dim=z_dim,
        feature_dims=feature_dims,
        **kwargs
    )

def get_default_loss_config(scenario):
    """
    Get default loss configuration for scenario.
    
    Args:
        scenario: Training scenario
        
    Returns:
        Loss configuration dictionary
    """
    base_config = {
        'ci': {'enabled': True, 'y_type': 'cont', 'detach_zm': True},
        'mbr': {'enabled': True, 'tau': 1.0, 'y_type': 'cont'},
        'mac': {'enabled': True, 'max_pairs': 4096},
        'align': {'enabled': True, 'temperature': 0.07},
        'style': {'enabled': False, 'style_type': 'regression', 'num_styles': 1},
        'ib': {'enabled': False, 'beta': 1e-4}
    }
    
    # Scenario-specific adjustments
    if scenario == 'IM':
        # Image as mediator - enable alignment for I_M <-> M_tab
        base_config['align']['enabled'] = True
    elif scenario == 'IY':
        # Image as outcome - enable style decoupling
        base_config['style']['enabled'] = True
    elif scenario == 'DUAL':
        # Both images - enable all components
        base_config['style']['enabled'] = True
        base_config['ib']['enabled'] = True
    
    return base_config
