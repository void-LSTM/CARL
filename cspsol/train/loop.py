"""
Training loop for CARL model.
Handles multi-phase training, gradient clipping, mixed precision, and monitoring.
Integrates CARL model with CSP data module and evaluation framework.
"""
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    # 按你的路径，parents[2] = /Volumes/Yulong/ICLR
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    __package__ = "cspsol.models"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import warnings
from collections import defaultdict
import numpy as np
from pathlib import Path
import json
import sys
from tqdm.auto import tqdm
import sys
from tqdm.auto import tqdm
from cspsol.train.sched import AdaptiveLRScheduler
from ..eval.hooks import RepresentationExtractor, CSPMetricsComputer
from ..models.carl import CausalAwareModel
from ..data.datamodule import CSPDataModule



class CSPTrainer:
    """
    Main trainer for CARL model with CSP datasets.
    
    Handles multi-phase training, automatic mixed precision,
    gradient clipping, and comprehensive monitoring.
    """
    
    def __init__(self,
                 model: CausalAwareModel,
                 datamodule: CSPDataModule,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 training_config: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize CSP trainer.
        
        Args:
            model: CARL model instance
            datamodule: CSP data module
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration
            training_config: Training configuration
            device: Training device
            output_dir: Output directory for logs and checkpoints
        """
        self.model = model
        self.datamodule = datamodule
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set default configurations
        self.optimizer_config = self._get_default_optimizer_config()
        if optimizer_config:
            self.optimizer_config.update(optimizer_config)
            
        self.scheduler_config = self._get_default_scheduler_config()
        if scheduler_config:
            self.scheduler_config.update(scheduler_config)
            
        self.training_config = self._get_default_training_config()
        if training_config:
            self.training_config.update(training_config)
            if 'early_stopping_metric' in training_config and 'early_stopping_mode' not in training_config:
                metric_name = str(training_config['early_stopping_metric'])
                if metric_name.endswith('loss'):
                    self.training_config['early_stopping_mode'] = 'min'
                else:
                    self.training_config['early_stopping_mode'] = 'max'

        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path('./outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_training()

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = (
            float('-inf')
            if self.training_config.get('early_stopping_mode', 'max') == 'max'
            else float('inf')
        )
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        print(f"CSP Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Output directory: {self.output_dir}")
    
    def _get_default_optimizer_config(self) -> Dict[str, Any]:
        """Get default optimizer configuration."""
        return {
            'name': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
    
    def _get_default_scheduler_config(self) -> Dict[str, Any]:
        """Get default scheduler configuration."""
        return {
            'name': 'cosine',
            'warmup_epochs': 5,
            'min_lr': 1e-6,
            'eta_min': 1e-6
        }
    
    def _get_default_training_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'max_epochs': 100,
            'gradient_clip_val': 5.0,
            'gradient_clip_algorithm': 'norm',
            'use_amp': True,
            'accumulate_grad_batches': 1,
            'val_check_interval': 1.0,
            'check_val_every_n_epoch': 1,
            'log_every_n_steps': 50,
            'save_every_n_epochs': 10,
            'early_stopping_patience': 15,
            'early_stopping_metric': 'val_struct_score',
            'early_stopping_mode': 'max',
            'skip_backward_on_no_grad': False
        }
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        name = self.optimizer_config['name'].lower()
        
        if name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config['weight_decay'],
                betas=self.optimizer_config['betas'],
                eps=self.optimizer_config['eps']
            )
        elif name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config['weight_decay'],
                betas=self.optimizer_config['betas'],
                eps=self.optimizer_config['eps']
            )
        elif name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config['weight_decay'],
                momentum=self.optimizer_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        name = self.scheduler_config['name'].lower()
        
        if name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['max_epochs'],
                eta_min=self.scheduler_config['eta_min']
            )
        elif name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_config.get('step_size', 30),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )
        elif name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 10),
                min_lr=self.scheduler_config['min_lr']
            )
        elif name == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {name}")
        
        # Setup warmup
        self.warmup_epochs = self.scheduler_config['warmup_epochs']
        self.base_lr = self.optimizer_config['lr']
    
    def _setup_training(self):
        """Setup training utilities."""
        self.use_amp = self.training_config['use_amp'] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        self.gradient_clip_val = self.training_config['gradient_clip_val']
        self.gradient_clip_algorithm = self.training_config['gradient_clip_algorithm']
        self.accumulate_grad_batches = self.training_config['accumulate_grad_batches']
        
        # Early stopping
        self.early_stopping_patience = self.training_config['early_stopping_patience']
        self.early_stopping_metric = self.training_config['early_stopping_metric']
        self.early_stopping_mode = self.training_config['early_stopping_mode']
        self.patience_counter = 0
    
    def _apply_warmup(self, epoch: int):
        """Apply learning rate warmup."""
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
    
    def _clip_gradients(self):
        """Apply gradient clipping."""
        if self.gradient_clip_val > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            
            if self.gradient_clip_algorithm == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
            elif self.gradient_clip_algorithm == 'value':
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
    
    def _forward_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        """Execute forward pass with optional mixed precision."""
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        if self.use_amp:
            with autocast():
                outputs = self.model(batch, epoch=epoch)
        else:
            outputs = self.model(batch, epoch=epoch)
        
        return outputs
    
    def _backward_step(self, loss: torch.Tensor):
        """Execute backward pass with optional mixed precision."""
        if not loss.requires_grad:
            message = "Total loss has no gradient; check active losses."
            if self.training_config.get('skip_backward_on_no_grad', False):
                warnings.warn(message)
                return
            raise ValueError(message)
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """Execute optimizer step with optional mixed precision."""
        self._clip_gradients()
        
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Optional staged mediator release
        training_phases = getattr(self.model, 'training_phases', {}) or {}
        release_epoch = training_phases.get('mediator_release_epoch')
        release_settings = training_phases.get('mediator_release_settings', {}) or {}
        if release_epoch is not None:
            try:
                release_epoch_int = int(release_epoch)
            except (TypeError, ValueError):
                release_epoch_int = None
        else:
            release_epoch_int = None

        if release_epoch_int is not None and epoch >= release_epoch_int:
            encoder = getattr(self.model, 'encoders', None)
            if encoder is not None and getattr(encoder, 'force_mediator', None) and not getattr(encoder, '_mediator_released', False):
                encoder.force_mediator = release_settings.get('force_mediator', False)
                encoder.y_direct_scale = release_settings.get('direct_scale', getattr(encoder, 'y_direct_scale', 0.0))
                if 'mediator_scale' in release_settings:
                    encoder.y_mediator_scale = release_settings['mediator_scale']
                encoder._mediator_release_settings = release_settings
                encoder._mediator_released = True
                encoder._mediator_last_schedule_epoch = epoch
                print(f"[Mediator release] epoch {epoch}: force_mediator={encoder.force_mediator}, "
                      f"direct_scale={encoder.y_direct_scale}")

        encoder = getattr(self.model, 'encoders', None)
        if encoder is not None and getattr(encoder, '_mediator_released', False):
            settings = getattr(encoder, '_mediator_release_settings', {}) or {}
            inc = settings.get('scale_increment')
            max_scale = settings.get('max_scale')
            if inc is not None:
                try:
                    inc = float(inc)
                except (TypeError, ValueError):
                    inc = None
            if max_scale is not None:
                try:
                    max_scale = float(max_scale)
                except (TypeError, ValueError):
                    max_scale = None
            last_epoch = getattr(encoder, '_mediator_last_schedule_epoch', None)
            if inc and (last_epoch is None or epoch > last_epoch):
                current_scale = getattr(encoder, 'y_direct_scale', 0.0)
                target_scale = current_scale + inc
                if max_scale is not None:
                    target_scale = min(target_scale, max_scale)
                if target_scale > current_scale:
                    encoder.y_direct_scale = target_scale
                    encoder._mediator_last_schedule_epoch = epoch
                    print(f"[Mediator schedule] epoch {epoch}: updated direct_scale={encoder.y_direct_scale}")
        epoch_metrics = defaultdict(list)
        
        # Apply warmup
        self._apply_warmup(epoch)
        
        # Get data loader
        train_loader = self.datamodule.train_dataloader()
        
        disable_tqdm = not sys.stdout.isatty()
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Epoch {epoch} [train]", leave=False,
                          dynamic_ncols=True, disable=disable_tqdm)

        # Training loop
        for batch_idx, batch in train_pbar:
            # Zero gradients
            if batch_idx % self.accumulate_grad_batches == 0:
                self.optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = self._forward_step(batch, epoch)
                loss = outputs['total_loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulate_grad_batches
                
                # Backward pass
                self._backward_step(loss)
                
                # Optimizer step
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    self._optimizer_step()
                    
                    # Update model step
                    self.model.update_step(self.current_step)
                    self.current_step += 1
                
                # Collect metrics
                has_diff_loss = any(
                    torch.is_tensor(v) and v.requires_grad for v in outputs.values()
                )
                total_loss_tensor = outputs.get('total_loss')
                if has_diff_loss and torch.is_tensor(total_loss_tensor):
                    epoch_metrics['train_total_loss'].append(total_loss_tensor.item())
                for key, value in outputs.items():
                    if key == 'total_loss':
                        continue
                    if torch.is_tensor(value) and value.dim() == 0:
                        epoch_metrics[f'train_{key}'].append(value.item())
                
                # Log progress
                if batch_idx % self.training_config['log_every_n_steps'] == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    display_loss = loss.detach().item() * self.accumulate_grad_batches
                    postfix = {'loss': f"{display_loss:.4f}", 'lr': f"{current_lr:.2e}"}
                    if 'loss_weights' in outputs:
                        for name, value in outputs['loss_weights'].items():
                            postfix[f'w_{name}'] = f"{float(value):.3f}"
                    train_pbar.set_postfix(postfix, refresh=False)

            except Exception as e:
                train_pbar.write(f"Error in training step {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        train_pbar.close()
        
        # Compute epoch averages
        epoch_avg = {}
        for key, values in epoch_metrics.items():
            if values:
                epoch_avg[key] = np.mean(values)
        
        return epoch_avg
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        epoch_metrics = defaultdict(list)
        
        # Get data loader
        val_loader = self.datamodule.val_dataloader()
        
        with torch.no_grad():
            disable_tqdm = not sys.stdout.isatty()
            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                            desc=f"Epoch {epoch} [val]", leave=False,
                            dynamic_ncols=True, disable=disable_tqdm)
            for batch_idx, batch in val_pbar:
                try:
                    # Forward pass
                    outputs = self._forward_step(batch, epoch)
                    
                    # Collect metrics - but only track scalar values, not for backprop
                    for key, value in outputs.items():
                        if torch.is_tensor(value) and value.dim() == 0:
                            # Detach from computation graph to avoid gradient issues
                            epoch_metrics[f'val_{key}'].append(value.detach().item())

                    if (batch_idx % self.training_config['log_every_n_steps'] == 0 and
                            epoch_metrics.get('val_total_loss')):
                        postfix = {'loss': f"{np.mean(epoch_metrics['val_total_loss']):.4f}"}
                        val_pbar.set_postfix(postfix, refresh=False)

                except Exception as e:
                    val_pbar.write(f"Error in validation step {batch_idx}: {e}")
                    continue

            val_pbar.close()
        
        # Compute epoch averages
        epoch_avg = {}
        for key, values in epoch_metrics.items():
            if values:
                epoch_avg[key] = np.mean(values)
        
        return epoch_avg
    
    def compute_structural_metrics(self) -> Dict[str, float]:
        """Compute CIP, CSI, MBRI, and MAC metrics on validation data."""
        try:
            val_loader = self.datamodule.val_dataloader()
            extractor = RepresentationExtractor(self.model, self.device)
            representations = extractor.extract_representations(
                val_loader,
                representation_types=['z_T', 'z_M', 'z_Y', 'z_Y_direct', 'y_gate']
            )
            metrics_computer = CSPMetricsComputer(self.model.scenario)
            metrics = metrics_computer.compute_all_metrics(representations)
            results = {}

            if 'CIP' in metrics:
                results['CIP'] = metrics['CIP'].get('cip_score')

            if 'CSI' in metrics:
                csi_metrics = metrics['CSI']
                results['CSI'] = csi_metrics.get('csi_score')
                extra_keys = [
                    'structure_preference', 'r2_mediated', 'r2_direct', 'r2_m_only',
                    'structure_preference_rf', 'r2_mediated_rf', 'r2_direct_rf', 'r2_m_only_rf', 'r2_perm_rf',
                    'csi_score_linear', 'structure_preference_linear',
                    'r2_mediated_linear', 'r2_direct_linear', 'r2_m_only_linear', 'r2_perm_linear'
                ]
                for key in extra_keys:
                    value = csi_metrics.get(key)
                    if value is not None:
                        results[f'CSI_{key}'] = value

            if 'MBRI' in metrics:
                results['MBRI'] = metrics['MBRI'].get('mbri_score')

            if 'MAC' in metrics:
                results['MAC'] = metrics['MAC'].get('mac_score')

            if 'RIC' in metrics:
                ric_metrics = metrics['RIC']
                results['RIC_score'] = ric_metrics.get('ric_score')
                for component in ('treatment', 'mediator', 'outcome'):
                    comp_value = ric_metrics.get(component)
                    if comp_value is not None:
                        results[f'RIC_{component}'] = comp_value
                if ric_metrics.get('ric_components') is not None:
                    results['RIC_components'] = ric_metrics.get('ric_components')

            if 'PSR' in metrics:
                psr_metrics = metrics['PSR']
                results['PSR'] = psr_metrics.get('psr_score')
                if psr_metrics.get('interaction_information') is not None:
                    results['PSR_interaction'] = psr_metrics.get('interaction_information')

            if 'MNI' in metrics:
                mni_metrics = metrics['MNI']
                results['MNI'] = mni_metrics.get('mni_score')
                if mni_metrics.get('mi_conditional') is not None:
                    results['MNI_conditional'] = mni_metrics.get('mi_conditional')

            if 'Y_GATE' in metrics:
                results['Y_GATE_MEAN'] = metrics['Y_GATE'].get('mean')
                results['Y_GATE_STD'] = metrics['Y_GATE'].get('std')

            # Aggregate structural score across available metrics
            score_candidates = [
                results.get('CIP'),
                results.get('CSI'),
                results.get('MBRI'),
                results.get('MAC'),
                results.get('RIC_score'),
                results.get('PSR'),
                results.get('MNI'),
            ]
            valid_scores = [v for v in score_candidates if v is not None and not np.isnan(v)]
            if valid_scores:
                results['STRUCT_SCORE'] = float(np.mean(valid_scores))

            return results
        except Exception as e:
            print(f"Error computing structural metrics: {e}")
        return {}
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check early stopping condition."""
        if self.early_stopping_metric not in val_metrics:
            return False

        raw_metric = val_metrics[self.early_stopping_metric]
        if raw_metric is None:
            return False

        try:
            current_metric = float(raw_metric)
        except (TypeError, ValueError):
            return False

        if np.isnan(current_metric):
            return False

        improved = False
        if self.early_stopping_mode == 'min':
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                improved = True
        else:  # max
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                improved = True
        
        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.early_stopping_patience
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'current_step': self.current_step,
            'model_config': {
                'scenario': self.model.scenario,
                'z_dim': self.model.z_dim,
                'feature_dims': self.model.feature_dims
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        if load_optimizer:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_step = checkpoint.get('current_step', 0)
        default_best = (
            float('-inf')
            if self.training_config.get('early_stopping_mode', 'max') == 'max'
            else float('inf')
        )
        self.best_metric = checkpoint.get('best_metric', default_best)
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resumed from epoch {self.current_epoch}, step {self.current_step}")
    
    def fit(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history dictionary
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        print(f"\nStarting training for {self.training_config['max_epochs']} epochs...")
        print(f"Starting from epoch {start_epoch}")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.training_config['max_epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % self.training_config['check_val_every_n_epoch'] == 0:
                val_metrics = self.validate_epoch(epoch)
            else:
                val_metrics = {}

            # Structural metrics (CIP, CSI, MBRI, MAC)
            struct_metrics = self.compute_structural_metrics()
            for key, value in struct_metrics.items():
                val_metrics[f'val_{key.lower()}'] = value

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update learning rate scheduler
            if self.scheduler and epoch >= self.warmup_epochs:
                if isinstance(self.scheduler, (optim.lr_scheduler.ReduceLROnPlateau, AdaptiveLRScheduler)):
                    metric_for_scheduler = val_metrics.get('val_total_loss', train_metrics.get('train_total_loss', 0))
                    self.scheduler.step(metric_for_scheduler)
                else:
                    self.scheduler.step()
            
            # Store metrics
            for key, value in all_metrics.items():
                if key.startswith('train_'):
                    self.train_metrics[key].append(value)
                elif key.startswith('val_'):
                    self.val_metrics[key].append(value)
            
            # Check early stopping
            should_stop = self._check_early_stopping(val_metrics)
            is_best = self.patience_counter == 0 and val_metrics
            
            # Save checkpoint
            if epoch % self.training_config['save_every_n_epochs'] == 0 or is_best or should_stop:
                self.save_checkpoint(epoch, all_metrics, is_best)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch:3d} completed in {epoch_time:.2f}s")
            print(f"Train loss: {train_metrics.get('train_total_loss', 0):.4f}")
            if val_metrics:
                print(f"Val loss:   {val_metrics.get('val_total_loss', 0):.4f}")
            if struct_metrics:
                print(
                    "CIP: {CIP:.4f} | CSI: {CSI:.4f} | MBRI: {MBRI:.4f} | MAC: {MAC:.4f}".format(
                        CIP=struct_metrics.get('CIP', float('nan')),
                        CSI=struct_metrics.get('CSI', float('nan')),
                        MBRI=struct_metrics.get('MBRI', float('nan')),
                        MAC=struct_metrics.get('MAC', float('nan')),
                    )
                )
                csi_delta = struct_metrics.get('CSI_structure_preference')
                if csi_delta is not None:
                    print(f"  CSI Δ (r2_mediated - r2_direct): {csi_delta:.6f}")
                ric_score = struct_metrics.get('RIC_score')
                if ric_score is not None:
                    ric_parts = []
                    for label in ('treatment', 'mediator', 'outcome'):
                        comp_val = struct_metrics.get(f'RIC_{label}')
                        if comp_val is not None:
                            ric_parts.append(f"{label[:3]}={comp_val:.4f}")
                    ric_details = ' | '.join(ric_parts) if ric_parts else ''
                    suffix = f" ({ric_details})" if ric_details else ''
                    print(f"  RIC: {ric_score:.4f}{suffix}")
                psr_score = struct_metrics.get('PSR')
                if psr_score is not None:
                    interaction = struct_metrics.get('PSR_interaction')
                    interaction_str = f", interaction={interaction:.4f}" if interaction is not None else ''
                    print(f"  PSR: {psr_score:.4f}{interaction_str}")
                mni_score = struct_metrics.get('MNI')
                if mni_score is not None:
                    mi_cond = struct_metrics.get('MNI_conditional')
                    mi_cond_str = f", MI_cond={mi_cond:.4f}" if mi_cond is not None else ''
                    print(f"  MNI: {mni_score:.4f}{mi_cond_str}")
                struct_score = struct_metrics.get('STRUCT_SCORE')
                if struct_score is not None:
                    print(f"  Structural score: {struct_score:.4f}")
                gate_mean = struct_metrics.get('Y_GATE_MEAN')
                gate_std = struct_metrics.get('Y_GATE_STD')
                if gate_mean is not None and gate_std is not None:
                    print(f"  y_gate mean={gate_mean:.4f} std={gate_std:.4f}")
            print(f"Best metric: {self.best_metric:.4f} (patience: {self.patience_counter}/{self.early_stopping_patience})")
            
            # Early stopping
            if should_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        training_time = time.time() - training_start_time
        print(f"\nTraining completed in {training_time:.2f}s ({training_time/3600:.2f}h)")
        
        # Save final metrics
        self._save_training_history()
        
        return {'train': dict(self.train_metrics), 'val': dict(self.val_metrics)}
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_metrics': {k: v for k, v in self.train_metrics.items()},
            'val_metrics': {k: v for k, v in self.val_metrics.items()},
            'best_metric': self.best_metric,
            'total_steps': self.current_step,
            'total_epochs': self.current_epoch + 1
        }
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {history_path}")


# Test implementation
if __name__ == "__main__":
    print("=== Testing CSP Trainer ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test scenario
    try:
        # Mock data directory - in real usage this would be actual CSP data
        print("Setting up test data module...")
        
        # Create dummy feature dimensions
        feature_dims = {
            'T_dim': 1, 'M_dim': 1, 'Y_dim': 1,
            'img_channels': 1, 'img_height': 28, 'img_width': 28
        }
        
        # Create test model
        print("Creating test CARL model...")
        loss_config = {
            'ci': {'enabled': True},
            'mbr': {'enabled': True},
            'mac': {'enabled': True},
            'align': {'enabled': False},  # Disable to reduce complexity
            'style': {'enabled': False},
            'ib': {'enabled': False}
        }
        
        model = CausalAwareModel(
            scenario='IM',
            z_dim=32,  # Smaller for faster testing
            feature_dims=feature_dims,
            loss_config=loss_config
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create mock data module
        class MockDataModule:
            def __init__(self, batch_size=8, num_batches=5):
                self.batch_size = batch_size
                self.num_batches = num_batches
                
            def train_dataloader(self):
                return self._create_loader()
            
            def val_dataloader(self):
                return self._create_loader(num_batches=3)
            
            def _create_loader(self, num_batches=None):
                if num_batches is None:
                    num_batches = self.num_batches
                    
                batches = []
                for _ in range(num_batches):
                    batch = {
                        'T': torch.randn(self.batch_size),
                        'M': torch.randn(self.batch_size),
                        'Y_star': torch.randn(self.batch_size),
                        'I_M': torch.randn(self.batch_size, 1, 28, 28),
                        'a_M': torch.rand(self.batch_size),
                        'b_style': torch.rand(self.batch_size)
                    }
                    batches.append(batch)
                return batches
        
        datamodule = MockDataModule()
        
        # Test trainer initialization
        print("Initializing trainer...")
        trainer_config = {
            'max_epochs': 3,  # Short test
            'log_every_n_steps': 2,
            'use_amp': False,  # Disable for testing
            'save_every_n_epochs': 2,
            'early_stopping_patience': 5
        }
        
        trainer = CSPTrainer(
            model=model,
            datamodule=datamodule,
            training_config=trainer_config,
            device=device,
            output_dir='./test_outputs'
        )
        
        print("Trainer initialized successfully")
        
        # Test single epoch
        print("\nTesting single training epoch...")
        train_metrics = trainer.train_epoch(epoch=0)
        print(f"Train metrics: {[(k, f'{v:.4f}') for k, v in train_metrics.items()]}")
        
        print("\nTesting single validation epoch...")
        val_metrics = trainer.validate_epoch(epoch=0)
        print(f"Val metrics: {[(k, f'{v:.4f}') for k, v in val_metrics.items()]}")
        
        # Test checkpoint saving/loading
        print("\nTesting checkpoint operations...")
        test_metrics = {**train_metrics, **val_metrics}
        trainer.save_checkpoint(epoch=0, metrics=test_metrics, is_best=True)
        
        # Modify model state
        original_state = trainer.model.state_dict()
        for param in trainer.model.parameters():
            param.data.fill_(0.5)
        
        # Load checkpoint
        trainer.load_checkpoint('./test_outputs/best_checkpoint.pt')
        
        # Verify restoration
        loaded_state = trainer.model.state_dict()
        state_match = all(torch.allclose(original_state[k], loaded_state[k]) 
                         for k in original_state.keys())
        print(f"Checkpoint load/save test: {'✓ PASSED' if state_match else '✗ FAILED'}")
        
        # Test short training run
        print("\nTesting full training loop...")
        history = trainer.fit()
        
        print(f"Training completed!")
        train_loss_history = history['train'].get('train_total_loss')
        val_loss_history = history['val'].get('val_total_loss')
        if train_loss_history:
            print(f"Final train loss: {train_loss_history[-1]:.4f}")
            assert len(train_loss_history) == trainer_config['max_epochs']
        else:
            print("Final train loss: N/A")
        if val_loss_history:
            print(f"Final val loss: {val_loss_history[-1]:.4f}")
            assert len(val_loss_history) <= trainer_config['max_epochs']
        else:
            print("Final val loss: N/A")

        
        print("✓ All trainer tests passed")
        
        # Cleanup
        import shutil
        if Path('./test_outputs').exists():
            shutil.rmtree('./test_outputs')
        
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== CSP Trainer Test Complete ===")
