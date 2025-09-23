"""
Evaluation hooks for CARL model assessment.
Integrates with CSPBench metrics and provides comprehensive evaluation framework.
Supports representation extraction, causal structure validation, and performance analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from math import erf, sqrt
import numpy as np
from pathlib import Path
import json
import pickle
from collections import defaultdict
import warnings
from abc import ABC, abstractmethod

# CSP-specific imports (would normally import from cspbench)
try:
    # This would be the actual import in production
    # from cspbench.eval_metrics import compute_cip, compute_csi, compute_mbri, compute_mac
    pass
except ImportError:
    warnings.warn("CSPBench not available, using mock implementations")


class RepresentationExtractor:
    """
    Extracts learned representations from trained CARL models.
    Supports batch processing and multiple representation types.
    """
    
    def __init__(self, 
                 model,
                 device: Optional[torch.device] = None):
        """
        Initialize representation extractor.
        
        Args:
            model: Trained CARL model
            device: Device for computation
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_representations(self, 
                              dataloader,
                              representation_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract representations from dataset.
        
        Args:
            dataloader: Data loader for extraction
            representation_types: Types of representations to extract
            
        Returns:
            Dictionary of extracted representations
        """
        if representation_types is None:
            representation_types = ['z_T', 'z_M', 'z_Y']
        
        # Storage for representations
        representations = {rep_type: [] for rep_type in representation_types}
        
        # Storage for additional data
        metadata = {
            'T': [], 'M': [], 'Y_star': [],
            'a_M': [], 'a_Y': [], 'b_style': [],
            'subject_id': [], 'scenario': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass to get representations
                    outputs = self.model(batch)
                    
                    # Extract requested representations
                    for rep_type in representation_types:
                        if rep_type in outputs:
                            rep_data = outputs[rep_type].cpu().numpy()
                            representations[rep_type].append(rep_data)
                    
                    # Extract metadata
                    for key in metadata.keys():
                        if key in batch:
                            if torch.is_tensor(batch[key]):
                                metadata[key].append(batch[key].cpu().numpy())
                            else:
                                metadata[key].append([batch[key]] * len(batch.get('T', [0])))
                
                except Exception as e:
                    print(f"Warning: Error processing batch {batch_idx}: {e}")
                    continue
        
        # Concatenate all batches
        final_representations = {}
        for rep_type, rep_list in representations.items():
            if rep_list:
                final_representations[rep_type] = np.concatenate(rep_list, axis=0)
        
        # Concatenate metadata
        final_metadata = {}
        for key, value_list in metadata.items():
            if value_list:
                # Handle different data types
                try:
                    if all(isinstance(v, np.ndarray) for v in value_list):
                        final_metadata[key] = np.concatenate(value_list, axis=0)
                    else:
                        # Flatten list of lists
                        flattened = []
                        for item in value_list:
                            if isinstance(item, (list, np.ndarray)):
                                flattened.extend(item)
                            else:
                                flattened.append(item)
                        final_metadata[key] = np.array(flattened)
                except Exception as e:
                    print(f"Warning: Could not process metadata {key}: {e}")
        
        # Ensure critical arrays share the same length
        result = {**final_representations, **final_metadata}
        keys_to_align = ['z_M', 'z_T', 'Y_star']
        arrays = [result.get(k) for k in keys_to_align if result.get(k) is not None]
        if len(arrays) == len(keys_to_align):
            min_len = min(a.shape[0] for a in arrays)
            for k in keys_to_align:
                arr = result[k]
                if arr.shape[0] != min_len:
                    result[k] = arr[:min_len]

        return result
    
    def save_representations(self, 
                           representations: Dict[str, np.ndarray],
                           output_path: str,
                           format: str = 'npz'):
        """
        Save extracted representations to disk.
        
        Args:
            representations: Dictionary of representations
            output_path: Output file path
            format: Save format ('npz', 'pickle', 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            np.savez_compressed(output_path, **representations)
        elif format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(representations, f)
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in representations.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Representations saved to: {output_path}")


class CSPMetricsComputer:
    """
    Computes CSP-specific evaluation metrics.
    Implements CIP, CSI, MBRI, and MAC metrics for causal structure assessment.
    """
    
    def __init__(self, 
                 scenario: str,
                 significance_level: float = 0.05,
                 n_bootstrap: int = 1000,
                 csi_permutations: int = 3,
                 random_state: int = 42):
        """
        Initialize CSP metrics computer.
        
        Args:
            scenario: Model scenario ('IM', 'IY', 'DUAL')
            significance_level: Significance level for statistical tests
            n_bootstrap: Number of bootstrap samples for confidence intervals
        """
        self.scenario = scenario
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        self.csi_permutations = max(1, int(csi_permutations))
        self.rng = np.random.default_rng(random_state)
    
    def compute_cip(self, 
                    z_T: np.ndarray, 
                    z_M: np.ndarray, 
                    Y: np.ndarray) -> Dict[str, float]:
        """
        Compute Conditional Independence Preservation (CIP).
        Tests if T ⊥ Y | M in representation space.
        
        Args:
            z_T: T representations
            z_M: M representations  
            Y: Target values
            
        Returns:
            Dictionary with CIP metrics
        """
        try:
            # Mock implementation - replace with actual CSPBench call
            from scipy.stats import pearsonr
            
            # Partial correlation approach (simplified)
            # In practice, would use more sophisticated conditional independence tests
            
            # Compute residuals of T and Y after regressing on M
            if z_M.ndim == 1:
                z_M = z_M.reshape(-1, 1)
            if z_T.ndim == 1:
                z_T = z_T.reshape(-1, 1)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            
            # Simple linear regression for residuals
            from sklearn.linear_model import LinearRegression
            
            reg_T = LinearRegression().fit(z_M, z_T)
            reg_Y = LinearRegression().fit(z_M, Y)
            
            residual_T = z_T - reg_T.predict(z_M)
            residual_Y = Y - reg_Y.predict(z_M)
            
            # Compute correlation between residuals
            if residual_T.shape[1] == 1:
                residual_T = residual_T.flatten()
            if residual_Y.shape[1] == 1:
                residual_Y = residual_Y.flatten()
            
            # Use first dimension if multi-dimensional
            if residual_T.ndim > 1:
                residual_T = residual_T[:, 0]
            if residual_Y.ndim > 1:
                residual_Y = residual_Y[:, 0]
            
            corr, p_value = pearsonr(residual_T, residual_Y)
            
            # CIP score: 1 if independent (p > alpha), 0 otherwise
            cip_score = 1.0 if p_value > self.significance_level else 0.0
            
            return {
                'cip_score': cip_score,
                'p_value': p_value,
                'correlation': abs(corr),
                'n_samples': len(z_T)
            }
            
        except Exception as e:
            print(f"Warning: CIP computation failed: {e}")
            return {'cip_score': 0.0, 'p_value': 1.0, 'correlation': 0.0, 'n_samples': 0}
    
    def _ensure_2d(self, array: np.ndarray) -> np.ndarray:
        """Convert arbitrary shaped array into (n_samples, n_features)."""
        arr = np.asarray(array)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim > 2:
            # Flatten feature dimensions but keep sample axis intact
            return arr.reshape(arr.shape[0], -1)
        return arr

    def compute_csi(self, 
                    z_T: np.ndarray, 
                    z_M: np.ndarray, 
                    z_Y: np.ndarray) -> Dict[str, float]:
        """
        Compute Causal Structure Identifiability (CSI).
        Evaluates preservation of 3-node causal structure T → M → Y.
        
        Args:
            z_T: T representations
            z_M: M representations
            z_Y: Y representations
            
        Returns:
            Dictionary with CSI metrics
        """
        try:
            from sklearn.base import clone
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            from sklearn.model_selection import KFold
            from sklearn.ensemble import RandomForestRegressor

            z_T_2d = self._ensure_2d(z_T)
            z_M_2d = self._ensure_2d(z_M)
            z_Y_2d = self._ensure_2d(z_Y)

            n_samples = min(z_T_2d.shape[0], z_M_2d.shape[0], z_Y_2d.shape[0])
            if n_samples < 3:
                raise ValueError("CSI requires at least three samples for cross-validation")

            if not (z_T_2d.shape[0] == z_M_2d.shape[0] == z_Y_2d.shape[0]):
                warnings.warn(
                    "CSI inputs have inconsistent lengths; trimming to matched samples"
                )
                z_T_2d = z_T_2d[:n_samples]
                z_M_2d = z_M_2d[:n_samples]
                z_Y_2d = z_Y_2d[:n_samples]

            n_splits = min(3, n_samples)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.rng.integers(1 << 31))
            splits = list(kf.split(range(n_samples)))

            def safe_r2(y_true, y_pred):
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return float('nan')
                try:
                    if np.allclose(np.std(y_true, axis=0), 0):
                        return 0.0
                    return float(r2_score(y_true, y_pred, multioutput='variance_weighted'))
                except ValueError:
                    return float('nan')

            def two_stage_cv(z_T_data, z_M_data, stage1, stage2, direct_estimator, m_only_estimator, fold_indices):
                r2_mediated = []
                r2_direct = []
                r2_m_only = []
                for train_idx, test_idx in fold_indices:
                    train_idx = np.asarray(train_idx)
                    test_idx = np.asarray(test_idx)

                    T_train, T_test = z_T_data[train_idx], z_T_data[test_idx]
                    M_train, M_test = z_M_data[train_idx], z_M_data[test_idx]
                    Y_train, Y_test = z_Y_2d[train_idx], z_Y_2d[test_idx]

                    stage1_model = clone(stage1)
                    stage1_model.fit(T_train, M_train)
                    Mhat_train = stage1_model.predict(T_train)

                    stage2_model = clone(stage2)
                    stage2_model.fit(Mhat_train, Y_train)
                    Mhat_test = stage1_model.predict(T_test)
                    Y_pred_mediated = stage2_model.predict(Mhat_test)
                    r2_mediated.append(safe_r2(Y_test, Y_pred_mediated))

                    direct_model = clone(direct_estimator)
                    direct_model.fit(T_train, Y_train)
                    Y_pred_direct = direct_model.predict(T_test)
                    r2_direct.append(safe_r2(Y_test, Y_pred_direct))

                    m_only_model = clone(m_only_estimator)
                    m_only_model.fit(M_train, Y_train)
                    Y_pred_m_only = m_only_model.predict(M_test)
                    r2_m_only.append(safe_r2(Y_test, Y_pred_m_only))

                return {
                    'mediated': np.asarray(r2_mediated, dtype=np.float64),
                    'direct': np.asarray(r2_direct, dtype=np.float64),
                    'm_only': np.asarray(r2_m_only, dtype=np.float64)
                }

            # Linear diagnostic (two-stage regression)
            linear_stage1 = LinearRegression()
            linear_stage2 = LinearRegression()
            linear_direct = LinearRegression()
            linear_results = two_stage_cv(
                z_T_2d,
                z_M_2d,
                linear_stage1,
                linear_stage2,
                linear_direct,
                linear_stage2,
                splits
            )

            r2_mediated_linear = np.nanmean(linear_results['mediated'])
            r2_direct_linear = np.nanmean(linear_results['direct'])
            r2_m_only_linear = np.nanmean(linear_results['m_only'])
            improvement_linear = r2_mediated_linear - r2_direct_linear

            perm_linear_improvements = []
            for _ in range(self.csi_permutations):
                perm_idx = self.rng.permutation(n_samples)
                z_M_perm = z_M_2d[perm_idx]
                perm_results = two_stage_cv(
                    z_T_2d,
                    z_M_perm,
                    linear_stage1,
                    linear_stage2,
                    linear_direct,
                    linear_stage2,
                    splits
                )
                perm_r2_mediated = np.nanmean(perm_results['mediated'])
                perm_linear_improvements.append(perm_r2_mediated - r2_direct_linear)
            perm_linear_improvements = np.asarray(perm_linear_improvements, dtype=np.float64)
            if perm_linear_improvements.size == 0 or np.all(np.isnan(perm_linear_improvements)):
                csi_linear = 0.5
                r2_perm_linear = float('nan')
            else:
                valid_perm = perm_linear_improvements[~np.isnan(perm_linear_improvements)]
                if valid_perm.size == 0:
                    csi_linear = 0.5
                    r2_perm_linear = float('nan')
                else:
                    greater = np.sum(improvement_linear > valid_perm)
                    ties = np.sum(np.isclose(improvement_linear, valid_perm, atol=1e-8))
                    csi_linear = float((greater + 0.5 * ties + 1) / (valid_perm.size + 1))
                    r2_perm_linear = float(np.mean(valid_perm + r2_direct_linear))

            if np.isnan(improvement_linear) or (csi_linear == 0.0 and r2_mediated_linear <= r2_direct_linear):
                warning_msg = 'mediated path underperforms direct path in current linear diagnostic'
            else:
                warning_msg = None

            # Non-linear diagnostic using Random Forest (two-stage)
            rf_seed = int(self.rng.integers(1 << 31))
            rf_stage1 = RandomForestRegressor(n_estimators=64, random_state=rf_seed, n_jobs=-1)
            rf_stage2 = RandomForestRegressor(n_estimators=64, random_state=rf_seed + 1, n_jobs=-1)
            rf_direct = RandomForestRegressor(n_estimators=64, random_state=rf_seed + 2, n_jobs=-1)

            rf_results = two_stage_cv(
                z_T_2d,
                z_M_2d,
                rf_stage1,
                rf_stage2,
                rf_direct,
                rf_stage2,
                splits
            )

            r2_mediated_rf = np.nanmean(rf_results['mediated'])
            r2_direct_rf = np.nanmean(rf_results['direct'])
            r2_m_only_rf = np.nanmean(rf_results['m_only'])
            improvement_rf = r2_mediated_rf - r2_direct_rf

            perm_rf_improvements = []
            for _ in range(self.csi_permutations):
                perm_idx = self.rng.permutation(n_samples)
                z_M_perm = z_M_2d[perm_idx]
                perm_results = two_stage_cv(
                    z_T_2d,
                    z_M_perm,
                    rf_stage1,
                    rf_stage2,
                    rf_direct,
                    rf_stage2,
                    splits
                )
                perm_r2_mediated = np.nanmean(perm_results['mediated'])
                perm_rf_improvements.append(perm_r2_mediated - r2_direct_rf)
            perm_rf_improvements = np.asarray(perm_rf_improvements, dtype=np.float64)
            if perm_rf_improvements.size == 0 or np.all(np.isnan(perm_rf_improvements)):
                csi_rf = 0.5
                r2_perm_rf = float('nan')
            else:
                valid_perm_rf = perm_rf_improvements[~np.isnan(perm_rf_improvements)]
                if valid_perm_rf.size == 0:
                    csi_rf = 0.5
                    r2_perm_rf = float('nan')
                else:
                    greater_rf = np.sum(improvement_rf > valid_perm_rf)
                    ties_rf = np.sum(np.isclose(improvement_rf, valid_perm_rf, atol=1e-8))
                    csi_rf = float((greater_rf + 0.5 * ties_rf + 1) / (valid_perm_rf.size + 1))
                    r2_perm_rf = float(np.mean(valid_perm_rf + r2_direct_rf))

            diagnostics = {
                'csi_score': csi_rf,
                'csi_score_rf': csi_rf,
                'r2_mediated_rf': float(r2_mediated_rf),
                'r2_direct_rf': float(r2_direct_rf),
                'r2_m_only_rf': float(r2_m_only_rf),
                'structure_preference_rf': float(improvement_rf),
                'r2_perm_rf': float(r2_perm_rf) if not np.isnan(r2_perm_rf) else None,
                'csi_score_linear': csi_linear,
                'r2_mediated_linear': float(r2_mediated_linear),
                'r2_direct_linear': float(r2_direct_linear),
                'r2_m_only_linear': float(r2_m_only_linear),
                'structure_preference_linear': float(improvement_linear),
                'r2_perm_linear': float(r2_perm_linear) if not np.isnan(r2_perm_linear) else None,
                'features_T': z_T_2d.shape[1],
                'features_M': z_M_2d.shape[1],
                'features_Y': z_Y_2d.shape[1],
                'n_samples': int(n_samples),
                'cv_splits': n_splits,
                'perm_samples': self.csi_permutations
            }

            if warning_msg is not None:
                diagnostics['reason_linear'] = warning_msg
            if np.isnan(improvement_rf) or (csi_rf == 0.0 and r2_mediated_rf <= r2_direct_rf):
                diagnostics['reason_rf'] = (
                    'mediated path underperforms direct path in random-forest diagnostic'
                )

            return diagnostics
            
        except Exception as e:
            print(f"Warning: CSI computation failed: {e}")
            return {'csi_score': 0.0, 'r2_mediated': 0.0, 'r2_direct': 0.0, 'structure_preference': 0.0}
    
    def compute_mbri(self, 
                     z_M: np.ndarray, 
                     z_T: np.ndarray, 
                     Y: np.ndarray) -> Dict[str, float]:
        """
        Compute Markov Boundary Retention Index (MBRI).
        Evaluates how well M serves as Markov boundary for Y.
        
        Args:
            z_M: M representations
            z_T: T representations
            Y: Target values
            
        Returns:
            Dictionary with MBRI metrics
        """
        try:
            # Mock implementation - replace with actual CSPBench call
            from sklearn.metrics import mutual_info_score
            from sklearn.preprocessing import KBinsDiscretizer
            # Reduce dimensionality to 1D per sample if needed
            if z_M.ndim > 1:
                z_M = z_M[:, 0]
            if z_T.ndim > 1:
                z_T = z_T[:, 0]
            if Y.ndim > 1:
                Y = Y[:, 0]

            # Ensure equal lengths
            min_len = min(len(z_M), len(z_T), len(Y))
            z_M = z_M[:min_len]
            z_T = z_T[:min_len]
            Y = Y[:min_len]

            # Discretize for mutual information computation
            discretizer = KBinsDiscretizer(
                n_bins=10,
                encode='ordinal',
                strategy='uniform',
                subsample=None
            )
            
            # Ensure arrays have shape (n_samples, n_features)
            z_M = np.atleast_2d(z_M)
            z_T = np.atleast_2d(z_T)
            Y = np.asarray(Y).reshape(-1)

            if z_M.shape[0] != Y.shape[0]:
                z_M = z_M.T
            if z_T.shape[0] != Y.shape[0]:
                z_T = z_T.T

            z_M_discrete = discretizer.fit_transform(z_M)
            z_T_discrete = discretizer.fit_transform(z_T)
            Y_discrete = discretizer.fit_transform(Y.reshape(-1, 1)).reshape(-1)

            # Compute mutual information per feature and aggregate
            I_M_Y = np.mean([
                mutual_info_score(z_M_discrete[:, i], Y_discrete)
                for i in range(z_M_discrete.shape[1])
            ])
            I_T_Y = np.mean([
                mutual_info_score(z_T_discrete[:, i], Y_discrete)
                for i in range(z_T_discrete.shape[1])
            ])
            
            # MBRI: I(M;Y) / (I(M;Y) + I(T;Y|M))
            # Simplified as I(M;Y) / (I(M;Y) + I(T;Y))
            denominator = I_M_Y + I_T_Y + 1e-8
            mbri_score = I_M_Y / denominator
            
            return {
                'mbri_score': mbri_score,
                'mi_m_y': I_M_Y,
                'mi_t_y': I_T_Y,
                'boundary_strength': I_M_Y / (I_T_Y + 1e-8)
            }
            
        except Exception as e:
            print(f"Warning: MBRI computation failed: {e}")
            return {'mbri_score': 0.0, 'mi_m_y': 0.0, 'mi_t_y': 0.0, 'boundary_strength': 0.0}
    
    def compute_mac(self, 
                    z_img: np.ndarray, 
                    a_semantic: np.ndarray) -> Dict[str, float]:
        """
        Compute Monotonic Alignment Consistency (MAC).
        Evaluates correlation between semantic and representation distances.
        
        Args:
            z_img: Image representations
            a_semantic: Semantic amplitude values
            
        Returns:
            Dictionary with MAC metrics
        """
        try:
            # Mock implementation - replace with actual CSPBench call
            from scipy.stats import spearmanr
            from scipy.spatial.distance import pdist, squareform
            
            # Compute pairwise distances
            semantic_distances = pdist(a_semantic.reshape(-1, 1), metric='euclidean')
            repr_distances = pdist(z_img, metric='euclidean')
            
            # Compute Spearman correlation
            correlation, p_value = spearmanr(semantic_distances, repr_distances)
            
            # MAC score: positive correlation indicates good alignment
            mac_score = max(0, correlation)
            
            return {
                'mac_score': mac_score,
                'correlation': correlation,
                'p_value': p_value,
                'n_pairs': len(semantic_distances)
            }
            
        except Exception as e:
            print(f"Warning: MAC computation failed: {e}")
            return {'mac_score': 0.0, 'correlation': 0.0, 'p_value': 1.0, 'n_pairs': 0}
    
    def compute_all_metrics(self, representations: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Compute all CSP metrics for given representations.
        
        Args:
            representations: Dictionary of extracted representations
            
        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}
        
        # Extract common variables
        z_T = representations.get('z_T')
        z_M = representations.get('z_M')
        z_Y = representations.get('z_Y')
        y_gate = representations.get('y_gate')
        Y_star = representations.get('Y_star')
        
        # Scenario-specific image representations and semantics
        if self.scenario == 'IM':
            z_img = z_M  # In IM, z_M comes from image
            a_semantic = representations.get('a_M')
        elif self.scenario == 'IY':
            z_img = representations.get('z_I_Y')
            a_semantic = representations.get('a_Y')
        else:  # DUAL
            z_img = representations.get('z_I_Y', representations.get('z_I_M'))
            a_semantic = representations.get('a_Y', representations.get('a_M'))
        
        # Compute CIP if we have required data
        if z_T is not None and z_M is not None and Y_star is not None:
            metrics['CIP'] = self.compute_cip(z_T, z_M, Y_star)
        
        # Compute CSI if we have required data
        if z_T is not None and z_M is not None and z_Y is not None:
            metrics['CSI'] = self.compute_csi(z_T, z_M, z_Y)
        
        # Compute MBRI if we have required data
        if z_M is not None and z_T is not None and Y_star is not None:
            # Align array lengths to avoid sample mismatch
            min_len = min(len(z_M), len(z_T), len(Y_star))
            if not (len(z_M) == len(z_T) == len(Y_star)):
                warnings.warn(
                    "MBRI inputs have inconsistent lengths; trimming to match"  # noqa: E501
                )
            z_M_aligned = z_M[:min_len]
            z_T_aligned = z_T[:min_len]
            Y_aligned = Y_star[:min_len]
            metrics['MBRI'] = self.compute_mbri(z_M_aligned, z_T_aligned, Y_aligned)
        
        # Compute MAC if we have required data
        if z_img is not None and a_semantic is not None:
            metrics['MAC'] = self.compute_mac(z_img, a_semantic)

        if isinstance(y_gate, np.ndarray) and y_gate.size > 0:
            metrics['Y_GATE'] = {
                'mean': float(np.mean(y_gate)),
                'std': float(np.std(y_gate))
            }

        return metrics


class ModelEvaluator:
    """
    Comprehensive model evaluator that combines representation extraction and metrics computation.
    Provides end-to-end evaluation pipeline for CARL models.
    """
    
    def __init__(self,
                 model,
                 scenario: str,
                 device: Optional[torch.device] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained CARL model
            scenario: Model scenario
            device: Computation device
            output_dir: Directory for saving results
        """
        self.model = model
        self.scenario = scenario
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir) if output_dir else Path('./eval_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = RepresentationExtractor(model, device)
        self.metrics_computer = CSPMetricsComputer(scenario)
    
    def evaluate(self, 
                dataloader,
                save_representations: bool = True,
                save_metrics: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            dataloader: Data loader for evaluation
            save_representations: Whether to save extracted representations
            save_metrics: Whether to save computed metrics
            
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting evaluation for {self.scenario} scenario...")
        
        # Extract representations
        print("Extracting representations...")
        representations = self.extractor.extract_representations(dataloader)
        
        # Save representations if requested
        if save_representations:
            repr_path = self.output_dir / 'representations.npz'
            self.extractor.save_representations(representations, repr_path)
        
        # Compute metrics
        print("Computing CSP metrics...")
        metrics = self.metrics_computer.compute_all_metrics(representations)
        
        # Compute summary statistics
        summary = self._compute_summary(metrics, representations)
        
        # Combine results
        results = {
            'scenario': self.scenario,
            'metrics': metrics,
            'summary': summary,
            'representation_shapes': {k: v.shape for k, v in representations.items() if isinstance(v, np.ndarray)},
            'n_samples': len(representations.get('z_T', []))
        }
        
        # Save metrics if requested
        if save_metrics:
            metrics_path = self.output_dir / 'evaluation_results.json'
            self._save_results(results, metrics_path)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _compute_summary(self, 
                        metrics: Dict[str, Dict[str, float]], 
                        representations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {}
        
        # Extract main scores
        main_scores = {}
        for metric_name, metric_dict in metrics.items():
            score_key = f'{metric_name.lower()}_score'
            if score_key in metric_dict:
                main_scores[metric_name] = metric_dict[score_key]
        
        summary['main_scores'] = main_scores
        
        # Compute overall score (average of available metrics)
        if main_scores:
            summary['overall_score'] = np.mean(list(main_scores.values()))
        else:
            summary['overall_score'] = 0.0
        
        # Representation quality analysis
        repr_quality = {}
        for repr_name, repr_data in representations.items():
            if isinstance(repr_data, np.ndarray) and repr_name.startswith('z_'):
                repr_quality[repr_name] = {
                    'mean_norm': float(np.mean(np.linalg.norm(repr_data, axis=1))),
                    'std_norm': float(np.std(np.linalg.norm(repr_data, axis=1))),
                    'mean_activation': float(np.mean(repr_data)),
                    'std_activation': float(np.std(repr_data))
                }
        
        summary['representation_quality'] = repr_quality
        
        return summary
    
    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save evaluation results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Evaluation results saved to: {output_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*50}")
        print(f"EVALUATION SUMMARY - {results['scenario']} Scenario")
        print(f"{'='*50}")
        
        # Main metrics
        main_scores = results['summary']['main_scores']
        if main_scores:
            print("\nCSP Metrics:")
            for metric, score in main_scores.items():
                status = "✓" if score > 0.7 else "⚠" if score > 0.4 else "✗"
                print(f"  {metric:6s}: {score:.4f} {status}")
            
            overall = results['summary']['overall_score']
            print(f"\nOverall Score: {overall:.4f}")
        else:
            print("No CSP metrics computed")
        
        # Sample information
        n_samples = results['n_samples']
        print(f"\nEvaluation Statistics:")
        print(f"  Samples: {n_samples}")
        print(f"  Representations: {len(results['representation_shapes'])}")
        
        # Representation shapes
        print(f"\nRepresentation Shapes:")
        for name, shape in results['representation_shapes'].items():
            print(f"  {name}: {shape}")
        
        print(f"{'='*50}\n")


# Test implementation
if __name__ == "__main__":
    print("=== Testing CSP Evaluation Framework ===")
    
    # Create mock model and data for testing
    class MockCARLModel(nn.Module):
        def __init__(self, scenario='IM'):
            super().__init__()
            self.scenario = scenario
            self.z_dim = 32
            self.feature_dims = {'T_dim': 1, 'M_dim': 1, 'Y_dim': 1}
            
            # Simple linear layers for testing
            self.encoder_T = nn.Linear(1, 32)
            self.encoder_M = nn.Linear(1, 32)
            self.encoder_Y = nn.Linear(1, 32)
        
        def forward(self, batch):
            outputs = {}
            if 'T' in batch:
                outputs['z_T'] = self.encoder_T(batch['T'].unsqueeze(-1))
            if 'M' in batch:
                outputs['z_M'] = self.encoder_M(batch['M'].unsqueeze(-1))
            if 'Y_star' in batch:
                outputs['z_Y'] = self.encoder_Y(batch['Y_star'].unsqueeze(-1))
            return outputs
    
    # Create mock dataloader
    class MockDataLoader:
        def __init__(self, n_batches=5, batch_size=16):
            self.n_batches = n_batches
            self.batch_size = batch_size
        
        def __iter__(self):
            for i in range(self.n_batches):
                yield {
                    'T': torch.randn(self.batch_size),
                    'M': torch.randn(self.batch_size),
                    'Y_star': torch.randn(self.batch_size),
                    'a_M': torch.rand(self.batch_size),
                    'a_Y': torch.rand(self.batch_size),
                    'subject_id': torch.arange(self.batch_size) + i * self.batch_size,
                    'scenario': 'IM'
                }
    
    print("\n--- Testing RepresentationExtractor ---")
    try:
        model = MockCARLModel()
        dataloader = MockDataLoader()
        
        extractor = RepresentationExtractor(model)
        representations = extractor.extract_representations(dataloader)
        
        print("Extracted representations:")
        for key, value in representations.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Test saving
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = f"{temp_dir}/test_representations.npz"
            extractor.save_representations(representations, save_path)
        
        print("✓ RepresentationExtractor test passed")
        
    except Exception as e:
        print(f"✗ RepresentationExtractor test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing CSPMetricsComputer ---")
    try:
        metrics_computer = CSPMetricsComputer(scenario='IM')
        
        # Create mock data
        n_samples = 100
        z_T = np.random.randn(n_samples, 32)
        z_M = np.random.randn(n_samples, 32)
        z_Y = np.random.randn(n_samples, 32)
        Y_star = np.random.randn(n_samples)
        a_semantic = np.random.rand(n_samples)
        
        # Test individual metrics
        cip_result = metrics_computer.compute_cip(z_T, z_M, Y_star)
        print(f"CIP result: {cip_result}")
        
        csi_result = metrics_computer.compute_csi(z_T, z_M, z_Y)
        print(f"CSI result: {csi_result}")
        
        mbri_result = metrics_computer.compute_mbri(z_M, z_T, Y_star)
        print(f"MBRI result: {mbri_result}")
        
        mac_result = metrics_computer.compute_mac(z_M, a_semantic)
        print(f"MAC result: {mac_result}")
        
        # Test all metrics together
        test_representations = {
            'z_T': z_T, 'z_M': z_M, 'z_Y': z_Y,
            'Y_star': Y_star, 'a_M': a_semantic
        }
        
        all_metrics = metrics_computer.compute_all_metrics(test_representations)
        print(f"All metrics computed: {list(all_metrics.keys())}")
        
        print("✓ CSPMetricsComputer test passed")
        
    except Exception as e:
        print(f"✗ CSPMetricsComputer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing ModelEvaluator ---")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            model = MockCARLModel()
            dataloader = MockDataLoader()
            
            evaluator = ModelEvaluator(
                model=model,
                scenario='IM',
                output_dir=temp_dir
            )
            
            # Run full evaluation
            results = evaluator.evaluate(
                dataloader,
                save_representations=True,
                save_metrics=True
            )
            
            # Verify results structure
            assert 'scenario' in results
            assert 'metrics' in results
            assert 'summary' in results
            assert 'representation_shapes' in results
            
            print("✓ ModelEvaluator test passed")
        
    except Exception as e:
        print(f"✗ ModelEvaluator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing edge cases ---")
    try:
        # Test with incomplete data
        incomplete_representations = {
            'z_T': np.random.randn(50, 32),
            'Y_star': np.random.randn(50)
            # Missing z_M
        }
        
        metrics_computer = CSPMetricsComputer(scenario='IM')
        partial_metrics = metrics_computer.compute_all_metrics(incomplete_representations)
        
        print(f"Partial metrics computed: {list(partial_metrics.keys())}")
        
        # Test with empty data
        empty_representations = {}
        empty_metrics = metrics_computer.compute_all_metrics(empty_representations)
        print(f"Empty metrics: {empty_metrics}")
        
        print("✓ Edge cases test passed")
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
    
    print("\n=== CSP Evaluation Framework Test Complete ===")
