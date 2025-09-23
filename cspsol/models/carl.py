"""
CARL: Causal-Aware Representation Learning model for CSP.
Main model that combines encoders, losses, and training logic for IM/IY/DUAL scenarios.
Integrates GradNorm, GRL, and multi-phase training strategy.
"""
# --- allow running this file directly via `python .../cspgen/sanity.py`
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    # 按你的路径，parents[2] = /Volumes/Yulong/ICLR
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    __package__ = "cspsol.models"
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from collections import defaultdict

from .encoders import CSPEncoderModule
from .losses import LossCI, LossMBR, LossMAC, LossAlign, LossStyle, LossIB, LossDecorr
from .gradnorm import MultiTaskBalancer
from ..utils.grl import create_grl


class CausalAwareModel(nn.Module):
    """
    Main CARL model for causal structure preservation.
    
    Handles IM/IY/DUAL scenarios with configurable loss components,
    automatic loss balancing, and multi-phase training.
    """
    
    def __init__(self,
                 scenario: str,
                 z_dim: int = 64,
                 feature_dims: Optional[Dict[str, int]] = None,
                 loss_config: Optional[Dict[str, Any]] = None,
                 encoder_config: Optional[Dict[str, Dict]] = None,
                 balancer_config: Optional[Dict[str, Any]] = None,
                 grl_config: Optional[Dict[str, Any]] = None,
                 training_phases: Optional[Dict[str, Dict]] = None):
        """
        Initialize CARL model.
        
        Args:
            scenario: Training scenario ('IM', 'IY', 'DUAL')
            z_dim: Latent representation dimension
            feature_dims: Feature dimensions from data module
            loss_config: Loss function configurations
            encoder_config: Encoder configurations
            balancer_config: Loss balancing configurations
            grl_config: Gradient reversal configurations
            training_phases: Multi-phase training configurations
        """
        super().__init__()
        
        self.scenario = scenario
        self.z_dim = z_dim
        
        # Set default configurations
        # Set default configurations with deep merging
        self.loss_config = self._get_default_loss_config()
        if loss_config:
            self.loss_config = self._deep_merge_config(self.loss_config, loss_config)
            
        self.encoder_config = self._get_default_encoder_config()
        if encoder_config:
            self.encoder_config = self._deep_merge_config(self.encoder_config, encoder_config)
            
        self.balancer_config = self._get_default_balancer_config()
        if balancer_config:
            self.balancer_config = self._deep_merge_config(self.balancer_config, balancer_config)
            
        self.grl_config = self._get_default_grl_config()
        if grl_config:
            self.grl_config = self._deep_merge_config(self.grl_config, grl_config)
            
        self.training_phases = self._get_default_training_phases()
        if training_phases:
            self.training_phases = self._deep_merge_config(self.training_phases, training_phases)

        self.mediator_release_epoch = None
        self.mediator_release_settings = {}
        
        # Set default feature dimensions
        if feature_dims is None:
            feature_dims = {
                'T_dim': 1, 'M_dim': 1, 'Y_dim': 1,
                'img_channels': 1, 'img_height': 28, 'img_width': 28
            }
        self.feature_dims = feature_dims
        
        # Build model components
        self._build_encoders()
        self.adversarial_heads = nn.ModuleDict()
        self.adv_grls = nn.ModuleDict()
        self._build_losses()
        self._build_balancer()
        self._build_grl()
        self._setup_predictors()
        self._setup_predictors()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.current_phase = 'warmup1'
        
        print(f"CARL model initialized for scenario: {scenario}")
        print(f"Latent dimension: {z_dim}")
        print(f"Active losses: {list(self.losses.keys())}")
    
    def _get_default_loss_config(self) -> Dict[str, Any]:
        """Get default loss configuration."""
        return {
            'ci': {'enabled': True, 'y_type': 'cont', 'detach_zm': True},
            'mbr': {'enabled': True, 'tau': 1.0, 'y_type': 'cont'},
            'mac': {'enabled': True, 'max_pairs': 4096},
            'align': {'enabled': False, 'temperature': 0.07},
            'style': {'enabled': False, 'style_type': 'regression', 'num_styles': 1},
            'ib': {'enabled': False, 'beta': 1e-4},
            'decor': {'enabled': False, 'weight': 0.5},
            'adv_t': {'enabled': False, 'weight': 0.5, 'alpha': 1.0}
        }
    
    def _get_default_encoder_config(self) -> Dict[str, Dict]:
        """Get default encoder configuration."""
        return {
            'tabular': {'hidden_dims': [128, 128], 'dropout': 0.1},
            'image': {'architecture': 'small_cnn', 'dropout': 0.1},
            'y_fusion': {
                'enabled': True,
                'force_mediator': False,
                'direct_scale': 0.5,
                'direct_dropout': 0.2,
                'gate_sharpness': 2.0,
                'mediator_scale': 1.0
            }
        }
    
    def _get_default_balancer_config(self) -> Dict[str, Any]:
        """Get default loss balancing configuration."""
        return {
            'method': 'gradnorm',  # 'gradnorm', 'dwa', 'fixed'
            'alpha': 0.5,
            'update_freq': 50,
            'initial_weights': {
                'ci': 1.0, 'mbr': 1.0, 'mac': 0.5,
                'align': 0.2, 'style': 0.1, 'ib': 0.01
            }
        }
    
    def _get_default_grl_config(self) -> Dict[str, Any]:
        """Get default GRL configuration."""
        return {
            'enabled': False,
            'adaptive': True,
            'max_alpha': 1.0,
            'schedule': 'linear',
            'warmup_steps': 1000
        }
    
    def _get_default_training_phases(self) -> Dict[str, Dict]:
        """
        Get default training phase configuration.

        The warmup phase needs at least one differentiable loss to provide
        gradients for the encoders.  ``LossMAC`` is correlation based and
        internally detaches tensors, therefore it cannot drive learning by
        itself.  When other losses such as ``align`` are disabled we fall back
        to enabling the first available differentiable loss (``ci`` or ``mbr``)
        so that warmup1 never trains with a zero gradient.  Users can still
        override this behaviour by supplying ``training_phases`` in the model
        configuration.
        """

        # Start with optional losses that are useful for warmup if enabled
        warmup1_losses: List[str] = []
        if self.loss_config.get('align', {}).get('enabled'):
            warmup1_losses.append('align')
        if self.loss_config.get('mac', {}).get('enabled'):
            warmup1_losses.append('mac')

        # Ensure at least one differentiable loss (ci or mbr) is present
        differentiable_candidates = ['ci', 'mbr']
        if not any(l in warmup1_losses for l in differentiable_candidates):
            for cand in differentiable_candidates:
                if self.loss_config.get(cand, {}).get('enabled'):
                    warmup1_losses.append(cand)
                    break

        # Fallback – if still empty, include whatever losses are enabled
        if not warmup1_losses:
            warmup1_losses = [
                name for name, cfg in self.loss_config.items()
                if cfg.get('enabled')
            ]

        warmup2_losses = ['ci', 'mbr', 'mac', 'align', 'decor']
        full_losses = ['ci', 'mbr', 'mac', 'align', 'style', 'ib', 'decor']
        if self.loss_config.get('adv_t', {}).get('enabled'):
            warmup2_losses.append('adv_t')
            full_losses.append('adv_t')

        return {
            'warmup1': {
                'epochs': [0, 10],
                'enabled_losses': warmup1_losses,
                'use_grl': False,
                'use_vib': False
            },
            'warmup2': {
                'epochs': [10, 20],
                'enabled_losses': warmup2_losses,
                'use_grl': False,
                'use_vib': False
            },
            'full': {
                'epochs': [20, float('inf')],
                'enabled_losses': full_losses,
                'use_grl': True,
                'use_vib': True
            }
        }
    def _deep_merge_config(self, default_config: Dict, user_config: Dict) -> Dict:
        """
        Deep merge user configuration with default configuration.
        
        Args:
            default_config: Default configuration dictionary
            user_config: User-provided configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = default_config.copy()
        
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._deep_merge_config(merged[key], value)
            else:
                # Replace or add new key
                merged[key] = value
        
        return merged
    def _build_encoders(self):
        """Build encoder modules."""
        self.encoders = CSPEncoderModule(
            scenario=self.scenario,
            z_dim=self.z_dim,
            feature_dims=self.feature_dims,
            encoder_configs=self.encoder_config
        )
    
    def _build_losses(self):
        """Build loss function modules."""
        self.losses = nn.ModuleDict()
        
        # Conditional Independence Loss
        if self.loss_config['ci']['enabled']:
            self.losses['ci'] = LossCI(
                zt_dim=self.z_dim,
                zm_dim=self.z_dim,
                y_type=self.loss_config['ci']['y_type'],
                detach_zm=self.loss_config['ci']['detach_zm']
            )
        
        # Markov Boundary Retention Loss
        if self.loss_config['mbr']['enabled']:
            self.losses['mbr'] = LossMBR(
                zm_dim=self.z_dim,
                zt_dim=self.z_dim,
                y_type=self.loss_config['mbr']['y_type'],
                tau=self.loss_config['mbr']['tau']
            )
        
        # Monotonic Alignment Consistency Loss
        if self.loss_config['mac']['enabled']:
            self.losses['mac'] = LossMAC(
                max_pairs=self.loss_config['mac']['max_pairs']
            )
        
        # Alignment Loss
        if self.loss_config['align']['enabled']:
            self.losses['align'] = LossAlign(
                temperature=self.loss_config['align']['temperature']
            )
        
        # Style Decoupling Loss
        if self.loss_config['style']['enabled']:
            self.losses['style'] = LossStyle(
                z_dim=self.z_dim,
                num_styles=self.loss_config['style']['num_styles'],
                style_type=self.loss_config['style']['style_type']
            )
        
        # Information Bottleneck Loss
        if self.loss_config['ib']['enabled']:
            self.losses['ib'] = LossIB(
                beta=self.loss_config['ib']['beta']
            )

        if self.loss_config['decor']['enabled']:
            self.losses['decor'] = LossDecorr()

        if self.loss_config.get('adv_t', {}).get('enabled'):
            hidden_dim = int(self.loss_config['adv_t'].get('hidden_dim', max(16, self.z_dim // 2)))
            self.adversarial_heads['t'] = nn.Sequential(
                nn.Linear(self.z_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            )
            adv_cfg = self.loss_config['adv_t']
            if adv_cfg.get('adaptive', False):
                self.adv_grls['t'] = create_grl(
                    adaptive=True,
                    max_alpha=float(adv_cfg.get('alpha', 1.0)),
                    schedule=str(adv_cfg.get('schedule', 'linear')),
                    warmup_steps=int(adv_cfg.get('warmup_steps', 1000))
                )
            else:
                self.adv_grls['t'] = create_grl(alpha=float(adv_cfg.get('alpha', 1.0)))
            self.losses['adv_t'] = nn.Identity()

    def _setup_predictors(self):
        """Configure auxiliary predictors and fusion parameters."""
        fusion_cfg = self.encoder_config.get('y_fusion', {}) if isinstance(self.encoder_config, dict) else {}
        residual_dim = fusion_cfg.get('residual_dim', 16)
        residual_scale = fusion_cfg.get('residual_scale', 0.5)

        if isinstance(residual_dim, (int, float)):
            residual_dim = int(residual_dim)
        else:
            residual_dim = None

        if hasattr(self, 'encoders'):
            self.encoders.residual_dim = residual_dim
            self.encoders.residual_scale = float(residual_scale)

        if fusion_cfg.get('enable_cross_predictor', True):
            hidden = fusion_cfg.get('predictor_hidden', self.z_dim)
            hidden = max(8, int(hidden))
            self.predict_y_from_m = nn.Sequential(
                nn.Linear(self.z_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.z_dim)
            )
            if hasattr(self, 'encoders'):
                self.encoders.predict_y_from_m = self.predict_y_from_m
        else:
            self.predict_y_from_m = None
            if hasattr(self, 'encoders'):
                self.encoders.predict_y_from_m = None
    
    def _build_balancer(self):
        """Build loss balancing module."""
        enabled_losses = [name for name in self.losses.keys()]
        
        if len(enabled_losses) > 1:
            self.balancer = MultiTaskBalancer(
                loss_names=enabled_losses,
                method=self.balancer_config['method'],
                **{k: v for k, v in self.balancer_config.items() if k != 'method'}
            )
        else:
            self.balancer = None
    
    def _build_grl(self):
        """Build gradient reversal layer if needed."""
        if self.grl_config['enabled']:
            self.grl = create_grl(
                adaptive=self.grl_config['adaptive'],
                max_alpha=self.grl_config['max_alpha'],
                schedule=self.grl_config['schedule'],
                warmup_steps=self.grl_config['warmup_steps']
            )
        else:
            self.grl = None
    
    def _update_training_phase(self, epoch: int):
        """Update current training phase based on epoch."""
        self.current_epoch = epoch
        
        for phase_name, phase_config in self.training_phases.items():
            epoch_range = phase_config['epochs']
            if epoch_range[0] <= epoch < epoch_range[1]:
                if self.current_phase != phase_name:
                    self.current_phase = phase_name
                    print(f"Switched to training phase: {phase_name} at epoch {epoch}")
                break
    
    def _get_active_losses(self) -> List[str]:
        """Get list of losses active in current training phase.

        Raises:
            RuntimeError: If no differentiable loss is active in the current
                phase. ``LossMAC`` alone cannot provide gradients as it uses
                detached tensors.  At least one other loss must be active to
                enable training.
        """
        if self.current_phase in self.training_phases:
            enabled_losses = self.training_phases[self.current_phase]['enabled_losses']
            active = [name for name in enabled_losses if name in self.losses]
        else:
            active = list(self.losses.keys())
        # Verify at least one loss will propagate gradients.  MAC is excluded
        # because it detaches its inputs for the Spearman correlation
        differentiable = [l for l in active if l != 'mac']
        if not differentiable:
            raise RuntimeError(
                "No differentiable losses active. Please enable at least one of "
                "['ci', 'mbr', 'align', 'style', 'ib'] for training."
            )
        return active
    
    def forward(self, batch: Dict[str, torch.Tensor], epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CARL model.
        
        Args:
            batch: Input batch from data loader
            epoch: Current training epoch (for phase switching)
            
        Returns:
            Dictionary containing representations and loss information
        """
        
        # Update training phase only when epoch is provided
        if epoch is not None:
            self._update_training_phase(epoch)
        
        # Get representations from encoders
        representations = self.encoders(batch)
        
        # Compute losses
        loss_outputs = self._compute_losses(batch, representations)
        
        # Combine with representations
        outputs = {**representations, **loss_outputs}
        
        return outputs
    
    def _compute_losses(self, batch: Dict[str, torch.Tensor], 
                       representations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all active losses.
        
        Args:
            batch: Input batch
            representations: Encoded representations
            
        Returns:
            Dictionary of computed losses
        """
        active_losses = self._get_active_losses()
        computed_losses = {}
        loss_components = {}
        
        # Extract commonly used representations
        z_T = representations.get('z_T')
        z_M = representations.get('z_M')
        z_Y = representations.get('z_Y')
        # Debug: Check for None values
        if z_T is None:
            print(f"Warning: z_T is None in scenario {self.scenario}")
        if z_M is None:
            print(f"Warning: z_M is None in scenario {self.scenario}")
        
        # Get target variables based on scenario
        # Get target variables based on scenario
        

        # Get target variables based on scenario
        if self.scenario == 'IM':
            # In IM scenario, Y is Y_star, and z_M comes from image
            y_target = batch.get('Y_star')
            z_target_img = z_M  # z_M is actually z_I_M in IM scenario
            # Handle empty batch case
            T_tensor = batch.get('T')
            if T_tensor is not None and T_tensor.numel() > 0:
                a_target = batch.get('a_M', torch.zeros_like(T_tensor))
            else:
                a_target = batch.get('a_M', torch.tensor([], device=next(self.parameters()).device))
        elif self.scenario == 'IY':
            # In IY scenario, Y target is phi_IY, z_M is tabular
            phi_target = batch.get('phi_IY')
            if phi_target is not None:
                if phi_target.dim() > 1:
                    # Use mean of phi features for CI/MBR
                    y_target = torch.mean(phi_target, dim=1, keepdim=False)  # (batch_size, 3) -> (batch_size,)
                else:
                    y_target = phi_target
            else:
                y_target = batch.get('Y_star')
            z_target_img = representations.get('z_I_Y')
            # Handle empty batch case
            T_tensor = batch.get('T')
            if T_tensor is not None and T_tensor.numel() > 0:
                a_target = batch.get('a_Y', torch.zeros_like(T_tensor))
            else:
                a_target = batch.get('a_Y', torch.tensor([], device=next(self.parameters()).device))
        else:  # DUAL
            # In DUAL scenario, use both
            y_target = batch.get('Y_star')
            z_target_img = representations.get('z_I_Y', representations.get('z_I_M'))
            # Handle empty batch case
            T_tensor = batch.get('T')
            if T_tensor is not None and T_tensor.numel() > 0:
                a_target = batch.get('a_Y', batch.get('a_M', torch.zeros_like(T_tensor)))
            else:
                a_target = batch.get('a_Y', batch.get('a_M', torch.tensor([], device=next(self.parameters()).device)))

        # Ensure y_target is 1D for loss functions
        if y_target is not None and y_target.dim() > 1:
            if y_target.shape[1] == 1:
                y_target = y_target.squeeze(1)  # (batch_size, 1) -> (batch_size,)
            else:
                y_target = torch.mean(y_target, dim=1)  # (batch_size, d) -> (batch_size,)
        if 'mbr' in active_losses and z_T is not None and z_M is not None and y_target is not None:
            y_embedding = representations.get('z_Y')
            mbr_loss, mbr_comps = self.losses['mbr'](z_M, z_T, y_target, y_embedding=y_embedding)
            computed_losses['mbr'] = mbr_loss
            loss_components.update({f'mbr_{k}': v for k, v in mbr_comps.items()})
        
        if 'mac' in active_losses and z_target_img is not None and a_target is not None:
            computed_losses['mac'] = self.losses['mac'](z_target_img, a_target)
        
        if 'align' in active_losses:
            if self.scenario == 'IM' and 'z_M_tab' in representations:
                # Align image-based M with tabular M
                computed_losses['align'] = self.losses['align'](z_M, representations['z_M_tab'])
            elif self.scenario in ['IY', 'DUAL'] and z_target_img is not None and z_M is not None:
                # Align image with tabular M
                computed_losses['align'] = self.losses['align'](z_target_img, z_M)
        
        if 'style' in active_losses and z_target_img is not None:
            # Apply GRL if enabled and in appropriate phase
            if (self.grl is not None and 
                self.training_phases[self.current_phase].get('use_grl', False)):
                z_for_style = self.grl(z_target_img)
            else:
                z_for_style = z_target_img
            
            style_target = batch.get('b_style', torch.zeros(z_target_img.shape[0], device=z_target_img.device))
            computed_losses['style'] = self.losses['style'](z_for_style, style_target)
        
        if 'ib' in active_losses and self.training_phases[self.current_phase].get('use_vib', False):
            # Apply IB to all representations
            ib_loss = 0.0
            for rep_name, rep_tensor in representations.items():
                if rep_name.startswith('z_'):
                    ib_loss += self.losses['ib'](rep_tensor)
            computed_losses['ib'] = ib_loss
        # Compute individual losses
        
        if 'ci' in active_losses and z_T is not None and z_M is not None and y_target is not None:
            computed_losses['ci'] = self.losses['ci'](z_T, z_M, y_target)
        else:
            if 'ci' in active_losses:
                print(f"CI loss skipped: active={'ci' in active_losses}, z_T={z_T is not None}, z_M={z_M is not None}, y_target={y_target is not None}")

        if 'decor' in active_losses and z_T is not None and z_Y is not None:
            decor_weight = self.loss_config['decor'].get('weight', 1.0)
            computed_losses['decor'] = decor_weight * self.losses['decor'](z_T, z_Y)

        if 'adv_t' in active_losses and 't' in self.adversarial_heads and T_tensor is not None:
            adv_head = self.adversarial_heads['t']
            adv_input = z_Y
            grl = self.adv_grls['t'] if 't' in self.adv_grls else None
            if grl is not None:
                if hasattr(grl, 'update_alpha'):
                    grl.update_alpha(self.current_step)
                adv_input = grl(adv_input)
            pred_t = adv_head(adv_input).view(-1)
            t_target = T_tensor.view(-1).float()
            weight = self.loss_config['adv_t'].get('weight', 1.0)
            computed_losses['adv_t'] = weight * F.mse_loss(pred_t, t_target)

        # Balance losses if balancer is available
        # Balance losses if balancer is available (only during training)
        if self.balancer is not None and len(computed_losses) > 1 and self.training:
            # Update balancer weights
            shared_params = list(self.encoders.parameters())
            loss_weights = self.balancer.update_weights(
                computed_losses,
                shared_parameters=shared_params
            )
            
            # Compute weighted total loss
            total_loss = sum(loss_weights.get(name, 0.0) * loss_val 
                        for name, loss_val in computed_losses.items())
            
            return {
                'total_loss': total_loss,
                'loss_weights': loss_weights,
                **computed_losses,
                **loss_components
            }
        else:
            # Simple sum for validation or when no balancer
            if computed_losses:
                total_loss = sum(computed_losses.values())
            else:
                # Ensure device consistency for empty loss
                device = next(self.parameters()).device
                if self.training:
                    # During training, create a small loss from model parameters
                    dummy_loss = sum(p.sum() for p in self.parameters()) * 0.0
                    total_loss = dummy_loss
                else:
                    # During validation, just use a zero tensor
                    total_loss = torch.tensor(0.0, device=device)
            
            return {
                'total_loss': total_loss,
                **computed_losses,
                **loss_components
            }
    
    def update_step(self, step: int):
        """Update training step for adaptive components."""
        self.current_step = step
        
        # Update GRL alpha if adaptive
        if (self.grl is not None and 
            hasattr(self.grl, 'update_alpha')):
            self.grl.update_alpha(step)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'current_phase': self.current_phase,
            'active_losses': self._get_active_losses(),
        }
        
        # Add balancer statistics
        if self.balancer is not None:
            stats['balancer'] = self.balancer.get_statistics()
        
        # Add GRL statistics
        if self.grl is not None:
            if hasattr(self.grl, 'get_alpha'):
                stats['grl_alpha'] = self.grl.get_alpha()
            elif hasattr(self.grl, 'alpha'):
                stats['grl_alpha'] = self.grl.alpha
        
        return stats
    
    def set_phase(self, phase_name: str):
        """Manually set training phase."""
        if phase_name in self.training_phases:
            self.current_phase = phase_name
            print(f"Manually set training phase to: {phase_name}")
        else:
            warnings.warn(f"Unknown phase: {phase_name}")


# # Test implementation
# if __name__ == "__main__":
#     print("=== Testing CARL Model ===")
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Test parameters
#     batch_size = 16
#     z_dim = 64
    
#     feature_dims = {
#         'T_dim': 1, 'M_dim': 1, 'Y_dim': 1,
#         'img_channels': 1, 'img_height': 28, 'img_width': 28
#     }
    
#     print(f"Test parameters:")
#     print(f"  batch_size: {batch_size}")
#     print(f"  z_dim: {z_dim}")
#     print(f"  feature_dims: {feature_dims}")
    
#     # Test scenarios
#     scenarios = ['IM', 'IY', 'DUAL']
    
#     for scenario in scenarios:
#         print(f"\n--- Testing scenario: {scenario} ---")
        
#         try:
#             # Create model
#             loss_config = {
#                 'ci': {'enabled': True, 'y_type': 'cont', 'detach_zm': True},
#                 'mbr': {'enabled': True, 'tau': 1.0, 'y_type': 'cont'},
#                 'mac': {'enabled': True},
#                 'align': {'enabled': True},
#                 'style': {'enabled': True},
#                 'ib': {'enabled': True}
#             }
            
#             model = CausalAwareModel(
#                 scenario=scenario,
#                 z_dim=z_dim,
#                 feature_dims=feature_dims,
#                 loss_config=loss_config
#             ).to(device)
            
#             # Create mock batch
#             batch = {
#                 'T': torch.randn(batch_size, device=device),
#                 'M': torch.randn(batch_size, device=device),
#                 'Y_star': torch.randn(batch_size, device=device),
#                 'a_M': torch.rand(batch_size, device=device),
#                 'a_Y': torch.rand(batch_size, device=device),
#                 'b_style': torch.rand(batch_size, device=device),
#                 'phi_IY': torch.randn(batch_size, 3, device=device)  # [brightness, contrast, texture]
#             }
            
#             # Add images based on scenario
#             if scenario in ['IM', 'DUAL']:
#                 batch['I_M'] = torch.randn(batch_size, 1, 28, 28, device=device)
#             else:
#                 batch['I_M'] = None
            
#             if scenario in ['IY', 'DUAL']:
#                 batch['I_Y'] = torch.randn(batch_size, 1, 28, 28, device=device)
#             else:
#                 batch['I_Y'] = None
            
#             print(f"Created model for {scenario} scenario")
#             print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
            
#             # Test different training phases
#             for epoch, phase in [(0, 'warmup1'), (15, 'warmup2'), (25, 'full')]:
#                 print(f"\n  Testing epoch {epoch} (phase: {phase})")
                
#                 # Forward pass
#                 outputs = model(batch, epoch=epoch)
                
#                 # Validate outputs
#                 assert 'total_loss' in outputs, "Should have total_loss"
#                 assert isinstance(outputs['total_loss'], torch.Tensor), "total_loss should be tensor"
#                 assert outputs['total_loss'].dim() == 0, "total_loss should be scalar"
                
#                 # Check representations
#                 repr_keys = [k for k in outputs.keys() if k.startswith('z_')]
#                 print(f"    Representations: {repr_keys}")
                
#                 for key in repr_keys:
#                     assert outputs[key].shape[0] == batch_size, f"{key} batch size mismatch"
#                     assert outputs[key].shape[1] == z_dim, f"{key} z_dim mismatch"
                
#                 # Check losses
#                 loss_keys = [k for k in outputs.keys() if k in ['ci', 'mbr', 'mac', 'align', 'style', 'ib']]
#                 print(f"    Active losses: {loss_keys}")
#                 print(f"    Total loss: {outputs['total_loss'].item():.4f}")
                
#                 # Check loss weights if available
#                 if 'loss_weights' in outputs:
#                     weights = outputs['loss_weights']
#                     print(f"    Loss weights: {weights}")
                
#                 # Update step for adaptive components
#                 model.update_step(epoch * 100)
            
#             # Test statistics
#             stats = model.get_statistics()
#             print(f"  Final statistics keys: {list(stats.keys())}")
#             print(f"  Current phase: {stats['current_phase']}")
#             print(f"  Active losses: {stats['active_losses']}")
            
#             print(f"✓ {scenario} scenario test passed")
            
#         except Exception as e:
#             print(f"✗ {scenario} scenario test failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     print("\n--- Testing custom configurations ---")
#     try:
#         # Test with custom configs
#         custom_loss_config = {
#             'ci': {'enabled': True, 'detach_zm': False},
#             'mbr': {'enabled': True, 'tau': 2.0},
#             'mac': {'enabled': False},
#             'align': {'enabled': False},
#             'style': {'enabled': False},
#             'ib': {'enabled': False}
#         }
        
#         custom_balancer_config = {
#             'method': 'dwa',
#             'temperature': 2.0,
#             'update_freq': 10
#         }
        
#         custom_grl_config = {
#             'enabled': True,
#             'adaptive': True,
#             'max_alpha': 2.0,
#             'schedule': 'exponential'
#         }
        
#         model = CausalAwareModel(
#             scenario='IY',
#             z_dim=32,
#             loss_config=custom_loss_config,
#             balancer_config=custom_balancer_config,
#             grl_config=custom_grl_config
#         ).to(device)
        
#         # Create minimal batch
#         batch = {
#             'T': torch.randn(8, device=device),
#             'M': torch.randn(8, device=device),
#             'Y_star': torch.randn(8, device=device),
#             'I_Y': torch.randn(8, 1, 28, 28, device=device),
#             'a_Y': torch.rand(8, device=device),
#             'b_style': torch.rand(8, device=device),
#             'phi_IY': torch.randn(8, 3, device=device)
#         }
        
#         outputs = model(batch, epoch=25)  # Full phase
        
#         print(f"Custom config test:")
#         print(f"  Total loss: {outputs['total_loss'].item():.4f}")
#         print(f"  Active losses: {[k for k in outputs.keys() if k in ['ci', 'mbr', 'mac']]}")
#         print(f"  z_dim: {outputs['z_T'].shape[1]}")
        
#         # Test manual phase setting
#         model.set_phase('warmup1')
#         outputs_warmup = model(batch)
#         print(f"  Manual phase change: {model.current_phase}")
        
#         print("✓ Custom configuration test passed")
        
#     except Exception as e:
#         print(f"✗ Custom configuration test failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n--- Testing edge cases ---")
#     try:
#         # Test with minimal losses
#         minimal_config = {
#             'ci': {'enabled': True},
#             'mbr': {'enabled': False},
#             'mac': {'enabled': False},
#             'align': {'enabled': False},
#             'style': {'enabled': False},
#             'ib': {'enabled': False}
#         }
        
#         print("Creating minimal model...")
#         model = CausalAwareModel(
#             scenario='IM',
#             loss_config=minimal_config
#         ).to(device)
        
#         # Set to full phase to enable all configured losses
#         model.set_phase('full')
        
#         print("Creating test batch...")
#         batch = {
#             'T': torch.randn(4, device=device),
#             'M': torch.randn(4, device=device),
#             'Y_star': torch.randn(4, device=device),
#             'I_M': torch.randn(4, 1, 28, 28, device=device)
#         }
        
#         print("Running forward pass...")
#         outputs = model(batch)
        
#         print("Validating outputs...")
#         assert 'total_loss' in outputs, "Missing total_loss"
#         assert 'ci' in outputs, "Missing ci loss"
#         print(f"Minimal config: loss={outputs['total_loss'].item():.4f}")
        
#         # Test empty batch handling
#         print("Testing empty batch...")
#         try:
#             empty_batch = {'T': torch.randn(0, device=device)}
#             outputs_empty = model(empty_batch)
#             print("Empty batch handled successfully")
#         except Exception as e:
#             print(f"Empty batch correctly rejected: {type(e).__name__}: {str(e)}")
        
#         # Test missing required fields
#         print("Testing incomplete batch...")
#         try:
#             incomplete_batch = {'T': torch.randn(4, device=device)}  # Missing required I_M for IM scenario
#             outputs_incomplete = model(incomplete_batch)
#             print("Incomplete batch handled successfully")
#         except Exception as e:
#             print(f"Incomplete batch correctly rejected: {type(e).__name__}: {str(e)}")
        
#         print("✓ Edge cases test passed")
        
#     except Exception as e:
#         print(f"✗ Edge cases test failed: {type(e).__name__}: {str(e)}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n=== CARL Model Test Complete ===")
