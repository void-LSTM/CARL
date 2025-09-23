"""
Encoder modules for CSP causal representation learning.
Supports both tabular and image encoders with flexible dimensions.
Designed for IM/IY scenarios and extensible to high-dimensional data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MLP(nn.Module):
    """
    Multi-layer perceptron with flexible architecture.
    Supports batch normalization, dropout, and various activations.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu',
                 use_batch_norm: bool = True,
                 dropout: float = 0.1,
                 final_activation: bool = False):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'gelu')
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            final_activation: Whether to apply activation to final layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if final_activation:
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(output_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class TabularEncoder(nn.Module):
    """
    Encoder for tabular data (T, M, Y_star, etc.).
    Handles both scalar and vector inputs flexibly.
    """
    
    def __init__(self,
                 input_dim: int,
                 z_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu',
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize tabular encoder.
        
        Args:
            input_dim: Input dimension (1 for scalar, >1 for vector)
            z_dim: Output latent dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        
        # Main encoder network
        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=z_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            final_activation=False
        )
        
        print(f"TabularEncoder initialized: {input_dim} -> {z_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch_size,) or (batch_size, input_dim)
        
        Returns:
            Encoded representation, shape (batch_size, z_dim)
        """
        # Handle scalar inputs
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (batch_size, 1)
        
        # Ensure correct input dimension
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[1]}")
        
        return self.encoder(x)


class ImageEncoder(nn.Module):
    """
    CNN encoder for images.
    Flexible architecture supporting different image sizes and channels.
    """
    
    def __init__(self,
                 z_dim: int,
                 img_channels: int = 1,
                 img_height: int = 28,
                 img_width: int = 28,
                 architecture: str = 'small_cnn',
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize image encoder.
        
        Args:
            z_dim: Output latent dimension
            img_channels: Number of input channels
            img_height: Image height
            img_width: Image width
            architecture: CNN architecture ('small_cnn', 'resnet_small')
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.architecture = architecture
        
        if architecture == 'small_cnn':
            self.encoder = self._build_small_cnn(use_batch_norm, dropout)
        elif architecture == 'resnet_small':
            self.encoder = self._build_resnet_small(use_batch_norm, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        print(f"ImageEncoder initialized: ({img_channels}, {img_height}, {img_width}) -> {z_dim}")
    
    def _build_small_cnn(self, use_batch_norm: bool, dropout: float) -> nn.Module:
        """Build small CNN for 28x28 images."""
        layers = []
        
        # First conv block: 1 -> 32
        layers.append(nn.Conv2d(self.img_channels, 32, kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))  # 28x28 -> 14x14
        
        # Second conv block: 32 -> 64
        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))  # 14x14 -> 7x7
        
        # Third conv block: 64 -> 128
        layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # 7x7 -> 1x1
        
        # Flatten and FC layers
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, self.z_dim))
        
        return nn.Sequential(*layers)
    
    def _build_resnet_small(self, use_batch_norm: bool, dropout: float) -> nn.Module:
        """Build small ResNet-style encoder."""
        # For future extension - basic implementation
        layers = []
        
        # Initial conv
        layers.append(nn.Conv2d(self.img_channels, 64, kernel_size=7, stride=2, padding=3))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, 2, 1))
        
        # Residual blocks would go here - simplified for now
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(64, self.z_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images, shape (batch_size, channels, height, width)
        
        Returns:
            Encoded representation, shape (batch_size, z_dim)
        """
        # Validate input shape
        expected_shape = (self.img_channels, self.img_height, self.img_width)
        if x.shape[1:] != expected_shape:
            raise ValueError(f"Expected image shape {expected_shape}, got {x.shape[1:]}")
        
        return self.encoder(x)


class FusionEncoder(nn.Module):
    """
    Multi-modal fusion encoder for combining tabular and image representations.
    Supports different fusion strategies.
    """
    
    def __init__(self,
                 input_dims: Dict[str, int],
                 z_dim: int,
                 fusion_method: str = 'concat',
                 hidden_dims: List[int] = [128],
                 activation: str = 'relu',
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize fusion encoder.
        
        Args:
            input_dims: Dictionary of input dimensions {'modality': dim}
            z_dim: Output latent dimension
            fusion_method: Fusion method ('concat', 'add', 'attention')
            hidden_dims: Hidden layer dimensions for fusion network
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.z_dim = z_dim
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            total_input_dim = sum(input_dims.values())
            self.fusion_net = MLP(
                input_dim=total_input_dim,
                output_dim=z_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                final_activation=False
            )
        elif fusion_method == 'add':
            # All inputs must have same dimension
            unique_dims = set(input_dims.values())
            if len(unique_dims) != 1:
                raise ValueError("For 'add' fusion, all input dims must be equal")
            
            input_dim = list(unique_dims)[0]
            self.fusion_net = MLP(
                input_dim=input_dim,
                output_dim=z_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                final_activation=False
            )
        elif fusion_method == 'attention':
            # Attention-based fusion (simplified)
            self.attention_nets = nn.ModuleDict({
                modality: nn.Linear(dim, 1) 
                for modality, dim in input_dims.items()
            })
            
            input_dim = list(input_dims.values())[0]  # Assume all same dim for attention
            self.fusion_net = MLP(
                input_dim=input_dim,
                output_dim=z_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                final_activation=False
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        print(f"FusionEncoder initialized: {input_dims} -> {z_dim} (method: {fusion_method})")
    
    def forward(self, representations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for multi-modal fusion.
        
        Args:
            representations: Dictionary of modality representations
        
        Returns:
            Fused representation, shape (batch_size, z_dim)
        """
        if self.fusion_method == 'concat':
            # Concatenate all representations
            rep_list = [representations[modality] for modality in sorted(representations.keys())]
            fused_input = torch.cat(rep_list, dim=1)
            return self.fusion_net(fused_input)
        
        elif self.fusion_method == 'add':
            # Element-wise addition
            rep_list = [representations[modality] for modality in sorted(representations.keys())]
            fused_input = sum(rep_list)
            return self.fusion_net(fused_input)
        
        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            attention_weights = {}
            for modality, rep in representations.items():
                attention_weights[modality] = torch.softmax(
                    self.attention_nets[modality](rep), dim=1
                )
            
            # Weighted sum
            weighted_reps = [
                attention_weights[modality] * representations[modality]
                for modality in sorted(representations.keys())
            ]
            fused_input = sum(weighted_reps)
            return self.fusion_net(fused_input)


class CSPEncoderModule(nn.Module):
    """
    Complete encoder module for CSP scenarios.
    Handles IM/IY/DUAL scenarios with appropriate encoders.
    """
    
    def __init__(self,
                 scenario: str,
                 z_dim: int = 64,
                 feature_dims: Optional[Dict[str, int]] = None,
                 encoder_configs: Optional[Dict[str, Dict]] = None):
        """
        Initialize CSP encoder module.
        
        Args:
            scenario: Scenario type ('IM', 'IY', 'DUAL')
            z_dim: Latent representation dimension
            feature_dims: Feature dimensions from data module
            encoder_configs: Configuration for individual encoders
        """
        super().__init__()
        
        self.scenario = scenario
        self.z_dim = z_dim
        self._mediator_released = False
        
        # Set default feature dimensions
        if feature_dims is None:
            feature_dims = {
                'T_dim': 1, 'M_dim': 1, 'Y_dim': 1,
                'img_channels': 1, 'img_height': 28, 'img_width': 28
            }
        
        # Set default encoder configs
        if encoder_configs is None:
            encoder_configs = {
                'tabular': {'hidden_dims': [128, 128], 'dropout': 0.1},
                'image': {'architecture': 'small_cnn', 'dropout': 0.1}
            }
        
        # Build encoders based on scenario
        self._build_encoders(feature_dims, encoder_configs)
        
        print(f"CSPEncoderModule initialized for scenario: {scenario}")
    
    def _build_encoders(self, feature_dims: Dict[str, int], configs: Dict[str, Dict]):
        """Build encoders based on scenario requirements."""
        tab_config = configs['tabular']
        img_config = configs['image']
        
        # Always build T encoder
        self.encoder_T = TabularEncoder(
            input_dim=feature_dims['T_dim'],
            z_dim=self.z_dim,
            **tab_config
        )
        
        # Always build Y_star encoder
        self.encoder_Y = TabularEncoder(
            input_dim=feature_dims['Y_dim'],
            z_dim=self.z_dim,
            **tab_config
        )

        fusion_cfg = configs.get('y_fusion', {})
        self.enable_y_fusion = fusion_cfg.get('enabled', True)
        self.force_mediator = fusion_cfg.get('force_mediator', False)
        self.y_direct_scale = float(fusion_cfg.get('direct_scale', 0.5)) if self.enable_y_fusion else 1.0
        dropout_rate = float(fusion_cfg.get('direct_dropout', 0.2)) if self.enable_y_fusion else 0.0
        self.y_direct_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.y_gate_beta = float(fusion_cfg.get('gate_sharpness', 2.0))
        self.y_mediator_scale = float(fusion_cfg.get('mediator_scale', 1.0))

        if self.enable_y_fusion:
            self.mediator_to_y = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.z_dim, self.z_dim)
            )
            self.y_fusion_gate = nn.Sequential(
                nn.Linear(self.z_dim * 2, self.z_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.z_dim, self.z_dim)
            )
        else:
            self.mediator_to_y = None
            self.y_fusion_gate = None
        
        # Build M encoders based on scenario
        if self.scenario == 'IM':
            # In IM scenario, M representation comes from image
            self.encoder_I_M = ImageEncoder(
                z_dim=self.z_dim,
                img_channels=feature_dims['img_channels'],
                img_height=feature_dims['img_height'],
                img_width=feature_dims['img_width'],
                **img_config
            )
            # Optional: tabular M encoder for comparison/alignment
            self.encoder_M_tab = TabularEncoder(
                input_dim=feature_dims['M_dim'],
                z_dim=self.z_dim,
                **tab_config
            )
        else:
            # In IY/DUAL scenarios, M is always tabular
            self.encoder_M = TabularEncoder(
                input_dim=feature_dims['M_dim'],
                z_dim=self.z_dim,
                **tab_config
            )
        
        # Build additional image encoders based on scenario
        if self.scenario == 'DUAL':
            # In DUAL scenario, we need both image encoders
            if not hasattr(self, 'encoder_I_M'):  # If not created above
                self.encoder_I_M = ImageEncoder(
                    z_dim=self.z_dim,
                    img_channels=feature_dims['img_channels'],
                    img_height=feature_dims['img_height'],
                    img_width=feature_dims['img_width'],
                    **img_config
                )
            
            self.encoder_I_Y = ImageEncoder(
                z_dim=self.z_dim,
                img_channels=feature_dims['img_channels'],
                img_height=feature_dims['img_height'],
                img_width=feature_dims['img_width'],
                **img_config
            )
        
        elif self.scenario == 'IY':
            # In IY scenario, only need I_Y encoder
            self.encoder_I_Y = ImageEncoder(
                z_dim=self.z_dim,
                img_channels=feature_dims['img_channels'],
                img_height=feature_dims['img_height'],
                img_width=feature_dims['img_width'],
                **img_config
            )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all relevant encoders.
        
        Args:
            batch: Batch dictionary from data loader
        
        Returns:
            Dictionary of encoded representations
        """
        representations = {}
        
        # Always encode T and Y
        representations['z_T'] = self.encoder_T(batch['T'])
        if self.enable_y_fusion and not getattr(self, '_fusion_debug_printed', False):
            print(f"[Encoder] force_mediator={self.force_mediator} direct_scale={self.y_direct_scale}")
            self._fusion_debug_printed = True
        z_y_direct = self.encoder_Y(batch['Y_star'])
        residual_dim = getattr(self, 'residual_dim', 16)
        if residual_dim and 0 < residual_dim < self.z_dim:
            residual_idx = torch.arange(residual_dim, device=z_y_direct.device)
            residual_part = z_y_direct[:, residual_idx]
            residual_scale = getattr(self, 'residual_scale', 0.5)
            z_y_direct = z_y_direct * residual_scale
            z_y_direct[:, residual_idx] = residual_part
        if self.y_direct_dropout is not None and self.training:
            z_y_direct = self.y_direct_dropout(z_y_direct)
        z_y_direct = z_y_direct * self.y_direct_scale
        representations['z_Y_direct'] = z_y_direct
        
        # Handle M encoding based on scenario
        if self.scenario == 'IM':
            # M representation from image (primary) 
            if 'I_M' in batch and batch['I_M'] is not None:
                representations['z_M'] = self.encoder_I_M(batch['I_M'])
            else:
                raise ValueError("IM scenario requires I_M in batch")
            
            # 可选：保留表格M作为对比/对齐用途
            if hasattr(self, 'encoder_M_tab') and 'M' in batch:
                representations['z_M_tab'] = self.encoder_M_tab(batch['M'])
        else:
            # M representation from tabular data
            if 'M' in batch:
                representations['z_M'] = self.encoder_M(batch['M'])
            else:
                raise ValueError(f"{self.scenario} scenario requires M in batch")

        if hasattr(self, 'predict_y_from_m') and callable(self.predict_y_from_m) and 'z_M' in representations:
            bias = self.predict_y_from_m(representations['z_M'])
        else:
            bias = 0.0

        # Handle additional image encodings for non-IM scenarios
        if self.scenario == 'DUAL' and 'I_M' in batch and batch.get('I_M') is not None:
            representations['z_I_M'] = self.encoder_I_M(batch['I_M'])

        if self.scenario in ['IY', 'DUAL'] and 'I_Y' in batch and batch.get('I_Y') is not None:
            representations['z_I_Y'] = self.encoder_I_Y(batch['I_Y'])

        if self.enable_y_fusion and 'z_M' in representations:
            mediator_influence = self.y_mediator_scale * self.mediator_to_y(representations['z_M']) + bias
            if self.force_mediator:
                z_y = mediator_influence
                representations['y_gate'] = torch.ones_like(mediator_influence)
            else:
                gate_input = torch.cat([z_y_direct, representations['z_M']], dim=1)
                gate_logits = self.y_fusion_gate(gate_input)
                gate = torch.sigmoid(self.y_gate_beta * gate_logits)
                z_y = (1 - gate) * z_y_direct + gate * mediator_influence
                representations['y_gate'] = gate
        else:
            z_y = z_y_direct + bias
            if self.enable_y_fusion:
                representations['y_gate'] = None

        representations['z_Y'] = z_y

        return representations
    
    def get_encoder(self, name: str) -> nn.Module:
        """Get specific encoder by name."""
        encoder_map = {
            'T': 'encoder_T',
            'Y': 'encoder_Y',
        }
        
        # Add M encoder based on scenario
        if self.scenario == 'IM':
            encoder_map['M'] = 'encoder_I_M'  # M comes from image in IM scenario
            if hasattr(self, 'encoder_M_tab'):
                encoder_map['M_tab'] = 'encoder_M_tab'
        else:
            encoder_map['M'] = 'encoder_M'  # M comes from tabular in IY/DUAL
        
        # Add image encoders if they exist
        if hasattr(self, 'encoder_I_M'):
            encoder_map['I_M'] = 'encoder_I_M'
        if hasattr(self, 'encoder_I_Y'):
            encoder_map['I_Y'] = 'encoder_I_Y'
        
        if name not in encoder_map:
            raise KeyError(f"Encoder '{name}' not found. Available: {list(encoder_map.keys())}")
        
        encoder_attr_name = encoder_map[name]
        if not hasattr(self, encoder_attr_name):
            raise AttributeError(f"Encoder attribute '{encoder_attr_name}' not found")
        
        return getattr(self, encoder_attr_name)

# # Test the implementation
# if __name__ == "__main__":
#     print("=== Testing CSP Encoders ===")
    
#     # Test device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     print("\n--- Testing MLP ---")
#     try:
#         mlp = MLP(input_dim=10, output_dim=64, hidden_dims=[32, 32])
#         x = torch.randn(8, 10)
#         output = mlp(x)
#         print(f"MLP test: {x.shape} -> {output.shape}")
#         assert output.shape == (8, 64), f"Expected (8, 64), got {output.shape}"
#         print("✓ MLP test passed")
#     except Exception as e:
#         print(f"MLP test failed: {e}")
    
#     print("\n--- Testing TabularEncoder ---")
#     try:
#         # Test scalar input
#         tab_encoder = TabularEncoder(input_dim=1, z_dim=64)
#         x_scalar = torch.randn(8)  # Scalar per sample
#         output_scalar = tab_encoder(x_scalar)
#         print(f"Tabular scalar: {x_scalar.shape} -> {output_scalar.shape}")
#         assert output_scalar.shape == (8, 64), f"Expected (8, 64), got {output_scalar.shape}"
        
#         # Test vector input
#         tab_encoder_vec = TabularEncoder(input_dim=5, z_dim=64)
#         x_vector = torch.randn(8, 5)
#         output_vector = tab_encoder_vec(x_vector)
#         print(f"Tabular vector: {x_vector.shape} -> {output_vector.shape}")
#         assert output_vector.shape == (8, 64), f"Expected (8, 64), got {output_vector.shape}"
        
#         print("✓ TabularEncoder test passed")
#     except Exception as e:
#         print(f"TabularEncoder test failed: {e}")
    
#     print("\n--- Testing ImageEncoder ---")
#     try:
#         img_encoder = ImageEncoder(z_dim=64, img_channels=1, img_height=28, img_width=28)
#         x_img = torch.randn(8, 1, 28, 28)
#         output_img = img_encoder(x_img)
#         print(f"Image encoder: {x_img.shape} -> {output_img.shape}")
#         assert output_img.shape == (8, 64), f"Expected (8, 64), got {output_img.shape}"
#         print("✓ ImageEncoder test passed")
#     except Exception as e:
#         print(f"ImageEncoder test failed: {e}")
    
#     print("\n--- Testing FusionEncoder ---")
#     try:
#         fusion_encoder = FusionEncoder(
#             input_dims={'tab': 64, 'img': 64},
#             z_dim=64,
#             fusion_method='concat'
#         )
#         representations = {
#             'tab': torch.randn(8, 64),
#             'img': torch.randn(8, 64)
#         }
#         output_fusion = fusion_encoder(representations)
#         print(f"Fusion encoder: {[v.shape for v in representations.values()]} -> {output_fusion.shape}")
#         assert output_fusion.shape == (8, 64), f"Expected (8, 64), got {output_fusion.shape}"
#         print("✓ FusionEncoder test passed")
#     except Exception as e:
#         print(f"FusionEncoder test failed: {e}")
    
#     print("\n--- Testing CSPEncoderModule ---")
#     try:
#         # Test different scenarios
#         for scenario in ['IM', 'IY', 'DUAL']:
#             print(f"\nTesting scenario: {scenario}")
            
#             encoder_module = CSPEncoderModule(
#                 scenario=scenario,
#                 z_dim=64,
#                 feature_dims={
#                     'T_dim': 1, 'M_dim': 1, 'Y_dim': 1,
#                     'img_channels': 1, 'img_height': 28, 'img_width': 28
#                 }
#             )
            
#             # Create mock batch
#             batch = {
#                 'T': torch.randn(4),
#                 'M': torch.randn(4),
#                 'Y_star': torch.randn(4),
#                 'I_M': torch.randn(4, 1, 28, 28) if scenario in ['IM', 'DUAL'] else None,
#                 'I_Y': torch.randn(4, 1, 28, 28) if scenario in ['IY', 'DUAL'] else None
#             }
            
#             # Forward pass
#             representations = encoder_module(batch)
            
#             print(f"Scenario {scenario} representations:")
#             for key, value in representations.items():
#                 print(f"  {key}: {value.shape}")
#                 assert value.shape == (4, 64), f"Expected (4, 64), got {value.shape}"
            
#             # Test encoder retrieval
#             t_encoder = encoder_module.get_encoder('T')
#             print(f"Retrieved T encoder: {type(t_encoder).__name__}")
        
#         print("✓ CSPEncoderModule test passed")
#     except Exception as e:
#         print(f"CSPEncoderModule test failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n--- Testing with real CSP data format ---")
#     try:
#         # Simulate real data format from CSPDataModule
#         encoder_module = CSPEncoderModule(
#             scenario='DUAL',
#             z_dim=64
#         )
        
#         # Mock batch similar to real data
#         mock_batch = {
#             'T': torch.tensor([-1.0397, -0.5234, 0.1234, 0.8765], dtype=torch.float32),
#             'M': torch.tensor([-1.7905, -0.3456, 0.2341, 1.2345], dtype=torch.float32),
#             'Y_star': torch.tensor([-1.5952, 0.1234, -0.5678, 0.9876], dtype=torch.float32),
#             'I_M': torch.randn(4, 1, 28, 28, dtype=torch.float32),
#             'I_Y': torch.randn(4, 1, 28, 28, dtype=torch.float32)
#         }
        
#         representations = encoder_module(mock_batch)
#         print("Real data format test:")
#         for key, value in representations.items():
#             print(f"  {key}: shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}")
        
#         print("✓ Real data format test passed")
#     except Exception as e:
#         print(f"Real data format test failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n=== CSP Encoders Test Complete ===")
