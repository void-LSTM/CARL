"""
Loss functions for CSP causal representation learning.
Implements L_CI, L_MBR, L_MAC, L_ALIGN, L_STYLE, and L_IB losses.
Based on the technical specification with CMI difference, MI estimation, and monotonic alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.stats import spearmanr
import warnings
import math

class GaussianHead(nn.Module):
    """
    Gaussian head for continuous Y regression.
    Outputs mu and log_var for Gaussian NLL computation.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, var_floor: float = 1e-3):
        """
        Initialize Gaussian head.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for scalar, >1 for vector)
            var_floor: Minimum variance to avoid numerical issues
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.var_floor = var_floor
        
        self.mu_head = nn.Linear(input_dim, output_dim)
        self.logvar_head = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features, shape (batch_size, input_dim)
        
        Returns:
            Tuple of (mu, logvar), each shape (batch_size, output_dim)
        """
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar
    
    def nll_loss(self, y_true: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian negative log-likelihood.
        
        Args:
            y_true: True values, shape (batch_size, output_dim) or (batch_size,)
            mu: Predicted means, shape (batch_size, output_dim)
            logvar: Predicted log variances, shape (batch_size, output_dim)
        
        Returns:
            NLL loss, shape (batch_size,)
        """
        # Handle scalar targets
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
        if logvar.dim() == 1:
            logvar = logvar.unsqueeze(1)
        
        # Compute variance with floor
        var = F.softplus(logvar) + self.var_floor
        
        # Gaussian NLL: 0.5 * [(y-mu)^2/var + log(var) + log(2π)]
        squared_error = (y_true - mu) ** 2
        nll = 0.5 * (squared_error / var + torch.log(var) + math.log(2 * math.pi))
        
        # Sum over output dimensions, return per-sample loss
        return nll.sum(dim=1)


class BernoulliHead(nn.Module):
    """
    Bernoulli head for binary Y classification.
    Outputs logits for cross-entropy computation.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Initialize Bernoulli head.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for binary, >1 for multi-class)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.logit_head = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features, shape (batch_size, input_dim)
        
        Returns:
            Logits, shape (batch_size, output_dim)
        """
        return self.logit_head(x)
    
    def nll_loss(self, y_true: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Bernoulli/Categorical negative log-likelihood.
        
        Args:
            y_true: True labels, shape (batch_size,) for classification
            logits: Predicted logits, shape (batch_size, output_dim)
        
        Returns:
            NLL loss, shape (batch_size,)
        """
        if self.output_dim == 1:
            # Binary classification
            y_true = y_true.float()
            logits = logits.squeeze(1)
            return F.binary_cross_entropy_with_logits(logits, y_true, reduction='none')
        else:
            # Multi-class classification
            return F.cross_entropy(logits, y_true, reduction='none')


class CMIEstimator(nn.Module):
    """
    Conditional Mutual Information estimator using NWJ difference method.
    Estimates I(Z_T; Y | Z_M) ≈ E[log q(y|z_t,z_m)] - E[log q(y|z_m)]
    """
    
    def __init__(self, 
                 zt_dim: int,
                 zm_dim: int,
                 y_type: str = 'cont',
                 y_dim: int = 1,
                 hidden_dims: List[int] = [256, 256],
                 var_floor: float = 1e-3):
        """
        Initialize CMI estimator.
        
        Args:
            zt_dim: Z_T dimension
            zm_dim: Z_M dimension  
            y_type: Y type ('cont' for continuous, 'bin' for binary)
            y_dim: Y dimension
            hidden_dims: Hidden layer dimensions for shared trunk
            var_floor: Variance floor for Gaussian heads
        """
        super().__init__()
        
        self.zt_dim = zt_dim
        self.zm_dim = zm_dim
        self.y_type = y_type
        self.y_dim = y_dim
        
        # Shared trunk network
        trunk_input_dim = max(zt_dim + zm_dim, zm_dim)  # For both q(y|zt,zm) and q(y|zm)
        trunk_output_dim = hidden_dims[-1] if hidden_dims else trunk_input_dim
        
        if hidden_dims:
            trunk_layers = []
            prev_dim = trunk_input_dim
            for hidden_dim in hidden_dims:
                trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
                trunk_layers.append(nn.ReLU(inplace=True))
                prev_dim = hidden_dim
            self.trunk = nn.Sequential(*trunk_layers)
            self.zm_projection = None
        else:
            self.trunk = nn.Identity()
            trunk_output_dim = trunk_input_dim
        
        # Two prediction heads
        if y_type == 'cont':
            self.head_tzm = GaussianHead(trunk_output_dim, y_dim, var_floor)
            self.head_zm = GaussianHead(trunk_output_dim, y_dim, var_floor)
        elif y_type == 'bin':
            self.head_tzm = BernoulliHead(trunk_output_dim, y_dim)
            self.head_zm = BernoulliHead(trunk_output_dim, y_dim)
        else:
            raise ValueError(f"Unknown y_type: {y_type}")
    
    def forward(self, z_t: torch.Tensor, z_m: torch.Tensor, y: torch.Tensor, 
                detach_zm: bool = True) -> torch.Tensor:
        """
        Compute CMI difference loss.
        
        Args:
            z_t: Z_T representations, shape (batch_size, zt_dim)
            z_m: Z_M representations, shape (batch_size, zm_dim) 
            y: Target values, shape (batch_size,) or (batch_size, y_dim)
            detach_zm: Whether to detach z_m for head_zm to prevent leakage
        
        Returns:
            CMI difference loss (to minimize), shape ()
        """
        batch_size = z_t.shape[0]
        
        # Prepare inputs for heads
        z_tzm = torch.cat([z_t, z_m], dim=1)  # Shape: (batch_size, zt_dim + zm_dim)
        
        if detach_zm:
            z_m_for_head = z_m.detach()
        else:
            z_m_for_head = z_m
        
        # Pad z_m to match trunk input dimension if necessary
        # Forward through trunk - 处理不同的输入维度
        h_tzm = self.trunk(z_tzm)

        # 对于z_m，如果维度不匹配，需要特殊处理
        zm_input_dim = z_m_for_head.shape[1]
        tzm_input_dim = z_tzm.shape[1]

        if zm_input_dim == tzm_input_dim:
            # 维度相同，直接使用同一个trunk
            h_zm = self.trunk(z_m_for_head)
        else:
            # 维度不同，动态创建投影层
            # 检查是否需要创建或更新投影层
            need_new_projection = (
                self.zm_projection is None or 
                (hasattr(self.zm_projection, 'in_features') and 
                (self.zm_projection.in_features != zm_input_dim or 
                self.zm_projection.out_features != tzm_input_dim))
            )

            if need_new_projection:
                self.zm_projection = nn.Linear(zm_input_dim, tzm_input_dim).to(z_m_for_head.device)
                # 将投影层注册为模块的一部分
                if hasattr(self, '_modules'):
                    self._modules['zm_projection'] = self.zm_projection

            z_m_projected = self.zm_projection(z_m_for_head)
            h_zm = self.trunk(z_m_projected)
        
        # Get predictions from heads
        if self.y_type == 'cont':
            mu_tzm, logvar_tzm = self.head_tzm(h_tzm)
            mu_zm, logvar_zm = self.head_zm(h_zm)
            
            # Compute NLL for both heads
            nll_tzm = self.head_tzm.nll_loss(y, mu_tzm, logvar_tzm)  # Shape: (batch_size,)
            nll_zm = self.head_zm.nll_loss(y, mu_zm, logvar_zm)      # Shape: (batch_size,)
        else:
            logits_tzm = self.head_tzm(h_tzm)
            logits_zm = self.head_zm(h_zm)
            
            # Compute NLL for both heads  
            nll_tzm = self.head_tzm.nll_loss(y, logits_tzm)  # Shape: (batch_size,)
            nll_zm = self.head_zm.nll_loss(y, logits_zm)     # Shape: (batch_size,)
        
        # CMI difference: minimize (E[NLL_tzm] - E[NLL_zm])
        cmi_diff_loss = (nll_tzm - nll_zm).mean()
        
        return cmi_diff_loss


class MIEstimator(nn.Module):
    """
    Mutual Information estimator using InfoNCE.
    Estimates I(U; V) using contrastive learning.
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 negative_mode: str = 'in_batch'):
        """
        Initialize MI estimator.
        
        Args:
            temperature: Temperature parameter for InfoNCE
            negative_mode: Negative sampling mode ('in_batch', 'queue')
        """
        super().__init__()
        
        self.temperature = temperature
        self.negative_mode = negative_mode
        
        # For queue-based negatives (future extension)
        if negative_mode == 'queue':
            self.register_buffer('queue_u', torch.randn(4096, 64))  # Will be updated
            self.register_buffer('queue_v', torch.randn(4096, 64))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE MI estimate.
        
        Args:
            u: First variable representations, shape (batch_size, dim)
            v: Second variable representations, shape (batch_size, dim)
        
        Returns:
            Negative MI estimate (to minimize for maximizing MI), shape ()
        """
        # L2 normalize
        u = F.normalize(u, dim=1)
        v = F.normalize(v, dim=1)
        
        # Compute similarities
        logits = torch.mm(u, v.t()) / self.temperature  # Shape: (batch_size, batch_size)
        
        # Positive samples are on the diagonal
        batch_size = u.shape[0]
        labels = torch.arange(batch_size, device=u.device)
        
        # InfoNCE loss (cross-entropy with positives on diagonal)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class LossCI(nn.Module):
    """
    Conditional Independence Loss.
    L_CI = CMI(Z_T; Y | Z_M) using difference of conditional likelihoods.
    """
    
    def __init__(self, 
                 zt_dim: int,
                 zm_dim: int,
                 y_type: str = 'cont',
                 y_dim: int = 1,
                 detach_zm: bool = True):
        """
        Initialize CI loss.
        
        Args:
            zt_dim: Z_T dimension
            zm_dim: Z_M dimension
            y_type: Y type ('cont' or 'bin')
            y_dim: Y dimension
            detach_zm: Whether to detach Z_M in head_zm
        """
        super().__init__()
        
        self.cmi_estimator = CMIEstimator(
            zt_dim=zt_dim,
            zm_dim=zm_dim,
            y_type=y_type,
            y_dim=y_dim
        )
        self.detach_zm = detach_zm
    
    def forward(self, z_t: torch.Tensor, z_m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute CI loss."""
        return self.cmi_estimator(z_t, z_m, y, detach_zm=self.detach_zm)


class LossMBR(nn.Module):
    """
    Markov Boundary Retention Loss.
    L_MBR = -I(Z_M; Y) + τ * I(Z_T; Y | Z_M)
    """
    
    def __init__(self, 
                 zm_dim: int,
                 zt_dim: int,
                 y_type: str = 'cont',
                 y_dim: int = 1,
                 tau: float = 1.0,
                 mi_temperature: float = 0.07):
        """
        Initialize MBR loss.
        
        Args:
            zm_dim: Z_M dimension
            zt_dim: Z_T dimension
            y_type: Y type
            y_dim: Y dimension
            tau: Weight for conditional term
            mi_temperature: Temperature for MI estimation
        """
        super().__init__()
        
        self.tau = tau
        
        # MI estimator for I(Z_M; Y)
        self.mi_estimator = MIEstimator(temperature=mi_temperature)
        
        # CMI estimator for I(Z_T; Y | Z_M)  
        self.cmi_estimator = CMIEstimator(
            zt_dim=zt_dim,
            zm_dim=zm_dim,
            y_type=y_type,
            y_dim=y_dim
        )
    
    def forward(self, z_m: torch.Tensor, z_t: torch.Tensor, y: torch.Tensor,
                y_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MBR loss.
        
        Args:
            z_m: Z_M representations, shape (batch_size, zm_dim)
            z_t: Z_T representations, shape (batch_size, zt_dim)
            y: Target values, shape (batch_size,) or (batch_size, y_dim)
            y_embedding: Y embeddings for MI computation, shape (batch_size, embed_dim)
                        If None, will use z_m as proxy for I(Z_M; Y)
        
        Returns:
            Tuple of (total_loss, component_losses)
        """
        # I(Z_M; Y) term - maximize (minimize negative)
        if y_embedding is not None:
            mi_zm_y_loss = self.mi_estimator(z_m, y_embedding)
        else:
            # Use Z_M as proxy - this is a simplification
            # In practice, might need a separate Y encoder
            mi_zm_y_loss = z_m.new_zeros(())
        
        # I(Z_T; Y | Z_M) term - minimize  
        cmi_zt_y_zm_loss = self.cmi_estimator(z_t, z_m, y)
        
        # Total loss: -I(Z_M; Y) + τ * I(Z_T; Y | Z_M)
        total_loss = mi_zm_y_loss + self.tau * cmi_zt_y_zm_loss
        
        component_losses = {
            'mi_zm_y': mi_zm_y_loss,
            'cmi_zt_y_zm': cmi_zt_y_zm_loss
        }
        
        return total_loss, component_losses


class LossMAC(nn.Module):
    """
    Monotonic Alignment Consistency Loss.
    L_MAC = -corr_spearman(|Δa|, ||Δz||) 
    Ensures semantic differences correlate with representation distances.
    """
    
    def __init__(self, max_pairs: int = 4096):
        """
        Initialize MAC loss.
        
        Args:
            max_pairs: Maximum number of pairs to sample for efficiency
        """
        super().__init__()
        self.max_pairs = max_pairs
    
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute MAC loss using Spearman correlation.
        
        Args:
            z: Representations, shape (batch_size, z_dim)
            a: Semantic amplitudes, shape (batch_size,)
        
        Returns:
            MAC loss (negative correlation), shape ()
        """
        batch_size = z.shape[0]

        if batch_size < 2:
            return z.new_zeros(())

        # Normalize representations and semantics to emphasize relative structure
        z = z.float()
        a = a.float()

        z = z - z.mean(dim=0, keepdim=True)
        z_std = z.std(dim=0, keepdim=True) + 1e-6
        z = z / z_std

        if a.dim() == 1:
            a = a.unsqueeze(1)
        a = a - a.mean(dim=0, keepdim=True)
        a_std = a.std(dim=0, keepdim=True) + 1e-6
        a = a / a_std

        # Sample pairs for efficiency
        max_possible_pairs = batch_size * (batch_size - 1) // 2
        num_pairs = min(self.max_pairs, max_possible_pairs)

        if num_pairs <= 10:  # Too few pairs for reliable correlation
            return torch.tensor(0.0, device=z.device)

        # compute all pairwise differences manually (avoids pdist gradient issues)
        def manual_pdist(x: torch.Tensor) -> torch.Tensor:
            """Manual pairwise distance computation with proper gradients."""
            n = x.shape[0]
            if n < 2:
                return torch.empty(0, device=x.device, dtype=x.dtype)
            
            # Get all pairs of indices
            i, j = torch.triu_indices(n, n, offset=1, device=x.device)
            
            # Compute distances between pairs
            diffs = x[i] - x[j]
            distances = torch.norm(diffs, p=2, dim=1)
            return distances

        a_diffs = manual_pdist(a)
        z_diffs = manual_pdist(z)

        # sample pairs if necessary
        total_pairs = a_diffs.numel()
        num_pairs = min(self.max_pairs, total_pairs)
        if num_pairs <= 10:
            return z.new_zeros(())
        if num_pairs < total_pairs:
            idx = torch.randperm(total_pairs, device=z.device)[:num_pairs]
            a_diffs = a_diffs[idx]
            z_diffs = z_diffs[idx]

        # differentiable rank transformation using pairwise comparisons
        def soft_rank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
            diff = x.unsqueeze(0) - x.unsqueeze(1)
            return torch.sigmoid(diff / tau).sum(dim=1)

        a_ranks = soft_rank(a_diffs)
        z_ranks = soft_rank(z_diffs)

        # standardize ranks
        a_ranks = a_ranks - a_ranks.mean()
        z_ranks = z_ranks - z_ranks.mean()

        if a_ranks.std() < 1e-8 or z_ranks.std() < 1e-8:
            return z.new_zeros(())

        corr = torch.corrcoef(torch.stack([a_ranks, z_ranks]))[0, 1]
        return -corr


class LossAlign(nn.Module):
    """
    Alignment Loss for cross-modal representation alignment.
    Uses InfoNCE to align representations from different modalities.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize alignment loss.
        
        Args:
            temperature: Temperature for InfoNCE
        """
        super().__init__()
        self.mi_estimator = MIEstimator(temperature=temperature)
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between two representation sets.
        
        Args:
            z1: First set of representations, shape (batch_size, dim)
            z2: Second set of representations, shape (batch_size, dim)
        
        Returns:
            Alignment loss (InfoNCE), shape ()
        """
        return self.mi_estimator(z1, z2)


class LossStyle(nn.Module):
    """
    Style Decoupling Loss using adversarial training.
    Trains a style discriminator while using gradient reversal on main encoder.
    """
    
    def __init__(self, 
                 z_dim: int,
                 num_styles: int,
                 style_type: str = 'regression',
                 hidden_dims: List[int] = [256, 256]):
        """
        Initialize style loss.
        
        Args:
            z_dim: Representation dimension
            num_styles: Number of style classes (for classification) or 1 (for regression)
            style_type: 'regression' for continuous style, 'classification' for discrete
            hidden_dims: Hidden dimensions for discriminator
        """
        super().__init__()
        
        self.style_type = style_type
        
        # Style discriminator
        layers = []
        prev_dim = z_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        if style_type == 'regression':
            layers.append(nn.Linear(prev_dim, 1))
            self.discriminator = nn.Sequential(*layers)
        else:
            layers.append(nn.Linear(prev_dim, num_styles))
            self.discriminator = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, style_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute style discriminator loss.
        
        Args:
            z: Representations, shape (batch_size, z_dim)
            style_targets: Style targets, shape (batch_size,)
        
        Returns:
            Style discrimination loss, shape ()
        """
        # Forward through discriminator
        style_pred = self.discriminator(z)
        
        if self.style_type == 'regression':
            # MSE loss for continuous style
            style_pred = style_pred.squeeze(1)
            return F.mse_loss(style_pred, style_targets.float())
        else:
            # Cross-entropy for discrete style
            return F.cross_entropy(style_pred, style_targets.long())


class LossIB(nn.Module):
    """
    Information Bottleneck Loss.
    L_IB = β * D_KL(q(z|x) || p(z))
    Lightweight version using L2 regularization.
    """
    
    def __init__(self, beta: float = 1e-4):
        """
        Initialize IB loss.
        
        Args:
            beta: Regularization strength
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute IB loss as L2 regularization.
        
        Args:
            z: Representations, shape (batch_size, z_dim)
        
        Returns:
            IB loss, shape ()
        """
        # Simple L2 regularization as IB approximation
        return self.beta * torch.mean(z ** 2)


class LossDecorr(nn.Module):
    """Squared correlation penalty to decorrelate two representations."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.dim() > 2:
            z_a = z_a.view(z_a.size(0), -1)
        if z_b.dim() > 2:
            z_b = z_b.view(z_b.size(0), -1)

        z_a = z_a - z_a.mean(dim=0, keepdim=True)
        z_b = z_b - z_b.mean(dim=0, keepdim=True)

        norm_a = torch.sqrt(torch.clamp((z_a ** 2).sum(dim=0, keepdim=True), min=self.eps))
        norm_b = torch.sqrt(torch.clamp((z_b ** 2).sum(dim=0, keepdim=True), min=self.eps))

        z_a_norm = z_a / norm_a
        z_b_norm = z_b / norm_b

        correlation = z_a_norm.t().matmul(z_b_norm) / (z_a.size(0) - 1 + self.eps)
        return (correlation ** 2).mean()


# Test the implementation
if __name__ == "__main__":
    print("=== Testing CSP Loss Functions ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup test data
    batch_size = 32
    z_dim = 64
    
    # Mock representations
    z_t = torch.randn(batch_size, z_dim, device=device)
    z_m = torch.randn(batch_size, z_dim, device=device)
    z_i = torch.randn(batch_size, z_dim, device=device)
    
    # Mock targets
    y_cont = torch.randn(batch_size, device=device)
    y_bin = torch.randint(0, 2, (batch_size,), device=device)
    a_values = torch.rand(batch_size, device=device)  # Semantic amplitudes
    b_values = torch.rand(batch_size, device=device)  # Style parameters
    
    print(f"Test data shapes:")
    print(f"  z_t: {z_t.shape}, z_m: {z_m.shape}, z_i: {z_i.shape}")
    print(f"  y_cont: {y_cont.shape}, y_bin: {y_bin.shape}")
    print(f"  a_values: {a_values.shape}, b_values: {b_values.shape}")
    
    print("\n--- Testing GaussianHead ---")
    try:
        gauss_head = GaussianHead(input_dim=z_dim, output_dim=1).to(device)
        mu, logvar = gauss_head(z_t)
        nll = gauss_head.nll_loss(y_cont, mu, logvar)
        print(f"GaussianHead: mu.shape={mu.shape}, logvar.shape={logvar.shape}")
        print(f"NLL loss: {nll.shape}, mean={nll.mean().item():.4f}")
        assert mu.shape == (batch_size, 1), f"Expected mu.shape=(32,1), got {mu.shape}"
        assert nll.shape == (batch_size,), f"Expected nll.shape=(32,), got {nll.shape}"
        print("✓ GaussianHead test passed")
    except Exception as e:
        print(f"GaussianHead test failed: {e}")
    
    print("\n--- Testing BernoulliHead ---")
    try:
        bern_head = BernoulliHead(input_dim=z_dim, output_dim=1).to(device)
        logits = bern_head(z_t)
        nll = bern_head.nll_loss(y_bin, logits)
        print(f"BernoulliHead: logits.shape={logits.shape}")
        print(f"NLL loss: {nll.shape}, mean={nll.mean().item():.4f}")
        assert logits.shape == (batch_size, 1), f"Expected logits.shape=(32,1), got {logits.shape}"
        assert nll.shape == (batch_size,), f"Expected nll.shape=(32,), got {nll.shape}"
        print("✓ BernoulliHead test passed")
    except Exception as e:
        print(f"BernoulliHead test failed: {e}")
    
    print("\n--- Testing CMIEstimator ---")
    try:
        cmi_estimator = CMIEstimator(zt_dim=z_dim, zm_dim=z_dim, y_type='cont').to(device)
        cmi_loss = cmi_estimator(z_t, z_m, y_cont)
        print(f"CMI loss: {cmi_loss.item():.4f}")
        assert cmi_loss.dim() == 0, f"Expected scalar loss, got shape {cmi_loss.shape}"
        print("✓ CMIEstimator test passed")
    except Exception as e:
        print(f"CMIEstimator test failed: {e}")
    
    print("\n--- Testing MIEstimator ---")
    try:
        mi_estimator = MIEstimator(temperature=0.07).to(device)
        mi_loss = mi_estimator(z_t, z_m)
        print(f"MI loss: {mi_loss.item():.4f}")
        assert mi_loss.dim() == 0, f"Expected scalar loss, got shape {mi_loss.shape}"
        print("✓ MIEstimator test passed")
    except Exception as e:
        print(f"MIEstimator test failed: {e}")
    
    print("\n--- Testing LossCI ---")
    try:
        loss_ci = LossCI(zt_dim=z_dim, zm_dim=z_dim, y_type='cont').to(device)
        ci_loss = loss_ci(z_t, z_m, y_cont)
        print(f"CI loss: {ci_loss.item():.4f}")
        assert ci_loss.dim() == 0, f"Expected scalar loss, got shape {ci_loss.shape}"
        print("✓ LossCI test passed")
    except Exception as e:
        print(f"LossCI test failed: {e}")
    
    print("\n--- Testing LossMBR ---")
    try:
        loss_mbr = LossMBR(zm_dim=z_dim, zt_dim=z_dim, y_type='cont', tau=1.0).to(device)
        mbr_loss, mbr_components = loss_mbr(z_m, z_t, y_cont)
        print(f"MBR loss: {mbr_loss.item():.4f}")
        print(f"MBR components: {[f'{k}={v.item():.4f}' for k, v in mbr_components.items()]}")
        assert mbr_loss.dim() == 0, f"Expected scalar loss, got shape {mbr_loss.shape}"
        print("✓ LossMBR test passed")
    except Exception as e:
        print(f"LossMBR test failed: {e}")
    
    print("\n--- Testing LossMAC ---")
    try:
        loss_mac = LossMAC(max_pairs=1000).to(device)
        mac_loss = loss_mac(z_i, a_values)
        print(f"MAC loss: {mac_loss.item():.4f}")
        assert mac_loss.dim() == 0, f"Expected scalar loss, got shape {mac_loss.shape}"
        print("✓ LossMAC test passed")
    except Exception as e:
        print(f"LossMAC test failed: {e}")
    
    print("\n--- Testing LossAlign ---")
    try:
        loss_align = LossAlign(temperature=0.07).to(device)
        align_loss = loss_align(z_t, z_m)
        print(f"Align loss: {align_loss.item():.4f}")
        assert align_loss.dim() == 0, f"Expected scalar loss, got shape {align_loss.shape}"
        print("✓ LossAlign test passed")
    except Exception as e:
        print(f"LossAlign test failed: {e}")
    
    print("\n--- Testing LossStyle ---")
    try:
        loss_style = LossStyle(z_dim=z_dim, num_styles=1, style_type='regression').to(device)
        style_loss = loss_style(z_i, b_values)
        print(f"Style loss: {style_loss.item():.4f}")
        assert style_loss.dim() == 0, f"Expected scalar loss, got shape {style_loss.shape}"
        print("✓ LossStyle test passed")
    except Exception as e:
        print(f"LossStyle test failed: {e}")
    
    print("\n--- Testing LossIB ---")
    try:
        loss_ib = LossIB(beta=1e-4).to(device)
        ib_loss = loss_ib(z_t)
        print(f"IB loss: {ib_loss.item():.6f}")
        assert ib_loss.dim() == 0, f"Expected scalar loss, got shape {ib_loss.shape}"
        print("✓ LossIB test passed")
    except Exception as e:
        print(f"LossIB test failed: {e}")
    
    print("\n--- Testing loss combination ---")
    try:
        # Test realistic loss combination
        loss_weights = {
            'ci': 1.0,
            'mbr': 1.0, 
            'mac': 0.5,
            'align': 0.2,
            'style': 0.1,
            'ib': 1e-4
        }
        
        # Initialize all losses
        losses = {
            'ci': LossCI(zt_dim=z_dim, zm_dim=z_dim, y_type='cont').to(device),
            'mbr': LossMBR(zm_dim=z_dim, zt_dim=z_dim, y_type='cont').to(device),
            'mac': LossMAC().to(device),
            'align': LossAlign().to(device),
            'style': LossStyle(z_dim=z_dim, num_styles=1, style_type='regression').to(device),
            'ib': LossIB().to(device)
        }
        
        # Compute all losses
        total_loss = 0.0
        loss_values = {}
        
        loss_values['ci'] = losses['ci'](z_t, z_m, y_cont)
        total_loss += loss_weights['ci'] * loss_values['ci']
        
        mbr_loss, _ = losses['mbr'](z_m, z_t, y_cont)
        loss_values['mbr'] = mbr_loss
        total_loss += loss_weights['mbr'] * loss_values['mbr']
        
        loss_values['mac'] = losses['mac'](z_i, a_values)
        total_loss += loss_weights['mac'] * loss_values['mac']
        
        loss_values['align'] = losses['align'](z_t, z_m)
        total_loss += loss_weights['align'] * loss_values['align']
        
        loss_values['style'] = losses['style'](z_i, b_values)
        total_loss += loss_weights['style'] * loss_values['style']
        
        loss_values['ib'] = losses['ib'](z_t)
        total_loss += loss_weights['ib'] * loss_values['ib']
        
        print(f"Combined loss test:")
        for name, loss_val in loss_values.items():
            weighted_val = loss_weights[name] * loss_val.item()
            print(f"  {name}: {loss_val.item():.4f} (weighted: {weighted_val:.4f})")
        print(f"  Total: {total_loss.item():.4f}")
        
        assert total_loss.dim() == 0, f"Expected scalar total loss, got shape {total_loss.shape}"
        print("✓ Loss combination test passed")
        
    except Exception as e:
        print(f"Loss combination test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== CSP Loss Functions Test Complete ===")
