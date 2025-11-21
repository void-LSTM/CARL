"""
Configuration management system for CARL experiments.
Provides unified configuration interface for models, training, data, and evaluation.
Supports YAML/JSON configs, parameter validation, and experiment tracking.
"""

import torch
import yaml
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict, field
from copy import deepcopy
import warnings
from datetime import datetime
import hashlib
import os


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    scenario: str = 'IM'
    data_dir: str = './data'
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    val_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42

    # Experimental factors for controlled comparisons
    dataset_size: Optional[int] = None
    noise_level: Optional[float] = None
    nonlinearity: Optional[str] = None
    image_complexity: Optional[str] = None
    
    # Data augmentation
    augmentation: Dict[str, Any] = field(default_factory=dict)
    
    # Preprocessing
    normalize: bool = True
    standardize: bool = False
    
    # Advanced options
    cache_data: bool = False
    preload_data: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.scenario not in ['IM', 'IY', 'DUAL']:
            raise ValueError(f"Invalid scenario: {self.scenario}")

        if not 0 < self.val_split < 1:
            raise ValueError(f"Invalid val_split: {self.val_split}")

        if not 0 < self.test_split < 1:
            raise ValueError(f"Invalid test_split: {self.test_split}")

        if self.val_split + self.test_split >= 1:
            raise ValueError("val_split + test_split must be < 1")

        if self.dataset_size is not None and self.dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive when provided: {self.dataset_size}")

        if self.noise_level is not None and not (0.0 <= self.noise_level <= 1.0):
            raise ValueError(f"noise_level must be in [0, 1]: {self.noise_level}")

        if self.nonlinearity is not None:
            allowed_nonlinearities = {'linear', 'quadratic', 'neural'}
            if self.nonlinearity not in allowed_nonlinearities:
                raise ValueError(
                    f"nonlinearity must be one of {sorted(allowed_nonlinearities)}: {self.nonlinearity}"
                )

        if self.image_complexity is not None:
            allowed_complexities = {'MNIST-style', 'CIFAR-style', 'Natural'}
            if self.image_complexity not in allowed_complexities:
                raise ValueError(
                    f"image_complexity must be one of {sorted(allowed_complexities)}: {self.image_complexity}"
                )


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    scenario: str = 'IM'
    z_dim: int = 128
    model_id: str = 'carl_full'
    baseline_config: Dict[str, Any] = field(default_factory=dict)

    # Feature dimensions (will be auto-detected if not provided)
    feature_dims: Optional[Dict[str, Any]] = None
    
    # Encoder configurations
    encoder_config: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dims': [256, 128],
        'activation': 'relu',
        'dropout': 0.1,
        'batch_norm': True
    })
    
    # Loss configuration
    loss_config: Dict[str, Any] = field(default_factory=lambda: {
        'ci': {'enabled': True, 'weight': 1.0},
        'mbr': {'enabled': True, 'weight': 1.0},
        'mac': {'enabled': True, 'weight': 0.5},
        'align': {'enabled': True, 'weight': 0.5},
        'style': {'enabled': False, 'weight': 0.1},
        'ib': {'enabled': False, 'weight': 0.1}
    })
    
    # Multi-task balancing
    balancer_config: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'gradnorm',
        'alpha': 1.5,
        'lr': 0.025
    })
    
    # Training phases
    phase_config: Dict[str, Any] = field(default_factory=lambda: {
        'warmup1_epochs': 20,
        'warmup2_epochs': 30,
        'warmup1_losses': ['mac', 'align'],
        'warmup2_losses': ['ci', 'mbr', 'mac', 'align']
    })
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.scenario not in ['IM', 'IY', 'DUAL']:
            raise ValueError(f"Invalid scenario: {self.scenario}")
        
        if self.z_dim <= 0:
            raise ValueError(f"z_dim must be positive: {self.z_dim}")

        if not self.model_id:
            raise ValueError("model_id must be specified in model configuration")


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = 'adamw'
    optimizer_config: Dict[str, Any] = field(default_factory=lambda: {
        'betas': (0.9, 0.999),
        'eps': 1e-8
    })
    
    # Learning rate scheduler
    scheduler: str = 'cosine_warmup_restarts'
    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {
        'warmup_epochs': 10,
        'first_cycle_steps': 50,
        'cycle_mult': 1.0,
        'min_lr': 1e-6
    })
    
    # Training dynamics
    gradient_clip_val: float = 5.0
    gradient_clip_algorithm: str = 'norm'
    accumulate_grad_batches: int = 1
    
    # Mixed precision
    use_amp: bool = True
    
    # Validation and checkpointing
    val_check_interval: float = 1.0
    check_val_every_n_epoch: int = 1
    save_every_n_epochs: int = 10
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = 'val_total_loss'
    early_stopping_mode: str = 'min'
    
    # Logging
    log_every_n_steps: int = 50
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive: {self.max_epochs}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive: {self.learning_rate}")
        
        if self.optimizer not in ['adam', 'adamw', 'sgd']:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Metrics to compute
    compute_cip: bool = True
    compute_csi: bool = True
    compute_mbri: bool = True
    compute_mac: bool = True
    
    # Evaluation parameters
    significance_level: float = 0.05
    n_bootstrap: int = 1000
    
    # Output options
    save_representations: bool = True
    save_metrics: bool = True
    save_plots: bool = True
    
    # Representation extraction
    representation_types: List[str] = field(default_factory=lambda: ['z_T', 'z_M', 'z_Y'])
    
    # Analysis options
    detailed_analysis: bool = True
    compare_baselines: bool = False
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        if not 0 < self.significance_level < 1:
            raise ValueError(f"Invalid significance_level: {self.significance_level}")
        
        if self.n_bootstrap <= 0:
            raise ValueError(f"n_bootstrap must be positive: {self.n_bootstrap}")


@dataclass
class RunConfig:
    """Configuration for a single training/evaluation run."""
    model_id: str = 'carl_full'
    variant: str = 'full'
    encoder_tag: str = 'shared_default'
    notes: str = ''
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model_id:
            raise ValueError("run.model_id must be specified")


@dataclass
class ExperimentGridConfig:
    """Configuration grid for systematic experiment sweeps."""
    data_scales: List[int] = field(default_factory=lambda: [500, 1000, 2000, 5000])
    noise_levels: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    nonlinearities: List[str] = field(default_factory=lambda: ['linear', 'quadratic', 'neural'])
    image_complexities: List[str] = field(default_factory=lambda: ['MNIST-style', 'CIFAR-style', 'Natural'])
    seeds: List[int] = field(default_factory=lambda: [0])

    def __post_init__(self):
        if not self.data_scales:
            raise ValueError("grid.data_scales must contain at least one value")
        if any(scale <= 0 for scale in self.data_scales):
            raise ValueError(f"All grid.data_scales must be positive: {self.data_scales}")

        if not self.noise_levels:
            raise ValueError("grid.noise_levels must contain at least one value")
        if any((level < 0.0 or level > 1.0) for level in self.noise_levels):
            raise ValueError(f"grid.noise_levels must lie in [0, 1]: {self.noise_levels}")

        allowed_nonlinearities = {'linear', 'quadratic', 'neural'}
        if not set(self.nonlinearities).issubset(allowed_nonlinearities):
            invalid = set(self.nonlinearities) - allowed_nonlinearities
            raise ValueError(f"grid.nonlinearities contains invalid entries: {sorted(invalid)}")

        allowed_complexities = {'MNIST-style', 'CIFAR-style', 'Natural'}
        if not set(self.image_complexities).issubset(allowed_complexities):
            invalid = set(self.image_complexities) - allowed_complexities
            raise ValueError(f"grid.image_complexities contains invalid entries: {sorted(invalid)}")

        if not self.seeds:
            raise ValueError("grid.seeds must contain at least one seed")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Meta information
    name: str = 'default_experiment'
    description: str = ''
    tags: List[str] = field(default_factory=list)
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Run selection and experiment matrix
    run: RunConfig = field(default_factory=RunConfig)
    grid: ExperimentGridConfig = field(default_factory=ExperimentGridConfig)
    
    # Experiment setup
    output_dir: str = './experiments'
    random_seed: int = 42
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False
    
    def __post_init__(self):
        """Validate complete configuration."""
        # Ensure scenario consistency
        if self.data.scenario != self.model.scenario:
            warnings.warn(f"Scenario mismatch: data={self.data.scenario}, model={self.model.scenario}")
        
        # Sync random seeds
        self.data.random_seed = self.random_seed
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID based on configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{timestamp}_{hash_obj.hexdigest()[:8]}"
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        # Convert tuples to lists for YAML compatibility
        config_dict = self._convert_tuples_to_lists(config_dict)
        
        if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
        print(f"Configuration saved to: {filepath}")

    def _convert_tuples_to_lists(self, obj):
        """Recursively convert tuples to lists for serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_tuples_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tuples_to_lists(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Convert lists back to tuples where appropriate
        config_dict = cls._convert_lists_to_tuples(config_dict)
        
        return cls.from_dict(config_dict)

    @classmethod
    def _convert_lists_to_tuples(cls, obj):
        """Convert specific lists back to tuples (for betas, etc.)."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k == 'betas' and isinstance(v, list) and len(v) == 2:
                    result[k] = tuple(v)
                else:
                    result[k] = cls._convert_lists_to_tuples(v)
            return result
        elif isinstance(obj, list):
            return [cls._convert_lists_to_tuples(item) for item in obj]
        else:
            return obj
    
    
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        # Create sub-configurations
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        run_config = RunConfig(**config_dict.get('run', {}))
        grid_config = ExperimentGridConfig(**config_dict.get('grid', {}))

        # Extract main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['data', 'model', 'training', 'evaluation', 'run', 'grid']}

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            run=run_config,
            grid=grid_config,
            **main_config
        )


class ConfigManager:
    """
    Manager for experiment configurations.
    Provides utilities for creating, validating, and managing configurations.
    """
    
    def __init__(self):
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, ExperimentConfig]:
        """Load preset configurations."""
        presets = {}
        
        # Quick development preset
        presets['dev'] = ExperimentConfig(
            name='development',
            description='Fast development configuration',
            data=DataConfig(
                batch_size=16,
                num_workers=2
            ),
            model=ModelConfig(
                z_dim=64,
                encoder_config={'hidden_dims': [128, 64]}
            ),
            training=TrainingConfig(
                max_epochs=10,
                learning_rate=1e-3,
                log_every_n_steps=5,
                save_every_n_epochs=5
            )
        )
        
        # High-quality research preset
        presets['research'] = ExperimentConfig(
            name='research',
            description='High-quality research configuration',
            data=DataConfig(
                batch_size=64,
                num_workers=8
            ),
            model=ModelConfig(
                z_dim=256,
                encoder_config={'hidden_dims': [512, 256]}
            ),
            training=TrainingConfig(
                max_epochs=200,
                learning_rate=1e-3,
                scheduler='cosine_warmup_restarts',
                early_stopping_patience=30
            ),
            evaluation=EvaluationConfig(
                n_bootstrap=5000,
                detailed_analysis=True
            )
        )
        
        # Production preset
        presets['production'] = ExperimentConfig(
            name='production',
            description='Production-ready configuration',
            data=DataConfig(
                batch_size=128,
                num_workers=16,
                cache_data=True
            ),
            model=ModelConfig(
                z_dim=512,
                encoder_config={'hidden_dims': [1024, 512, 256]}
            ),
            training=TrainingConfig(
                max_epochs=500,
                learning_rate=1e-3,
                use_amp=True,
                gradient_clip_val=10.0
            ),
            evaluation=EvaluationConfig(
                n_bootstrap=10000,
                detailed_analysis=True,
                compare_baselines=True
            )
        )
        
        return presets
    
    def get_preset(self, name: str) -> ExperimentConfig:
        """Get preset configuration by name."""
        if name not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        
        return deepcopy(self.presets[name])
    
    def create_config(self, 
                     preset: Optional[str] = None,
                     **overrides) -> ExperimentConfig:
        """
        Create configuration with optional preset and overrides.
        
        Args:
            preset: Preset name to start from
            **overrides: Configuration overrides
            
        Returns:
            ExperimentConfig instance
        """
        if preset:
            config = self.get_preset(preset)
        else:
            config = ExperimentConfig()
        
        # Apply overrides
        self._apply_overrides(config, overrides)
        
        return config
    
    def _apply_overrides(self, config: ExperimentConfig, overrides: Dict[str, Any]):
        """Apply configuration overrides with proper type handling."""
        for key, value in overrides.items():
            if '.' in key:
                # Nested key like 'model.z_dim'
                parts = key.split('.')
                obj = config
                
                # Navigate to the parent object
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                
                # Get the final attribute name
                final_attr = parts[-1]
                
                # Get current value to preserve type if possible
                try:
                    current_value = getattr(obj, final_attr)
                    # Try to convert to the same type as current value
                    if isinstance(current_value, bool):
                        converted_value = bool(value)
                    elif isinstance(current_value, int):
                        converted_value = int(value)
                    elif isinstance(current_value, float):
                        converted_value = float(value)
                    elif isinstance(current_value, str):
                        converted_value = str(value)
                    else:
                        converted_value = value
                except (AttributeError, ValueError, TypeError):
                    # If type conversion fails or attribute doesn't exist, use original value
                    converted_value = value
                
                setattr(obj, final_attr, converted_value)
                
            else:
                # Top-level key
                try:
                    current_value = getattr(config, key)
                    # Try to convert to the same type as current value
                    if isinstance(current_value, bool):
                        converted_value = bool(value)
                    elif isinstance(current_value, int):
                        converted_value = int(value)
                    elif isinstance(current_value, float):
                        converted_value = float(value)
                    elif isinstance(current_value, str):
                        converted_value = str(value)
                    else:
                        converted_value = value
                except (AttributeError, ValueError, TypeError):
                    # If type conversion fails or attribute doesn't exist, use original value
                    converted_value = value
                
                setattr(config, key, converted_value)
    
    def _estimate_memory_usage(self, config: ExperimentConfig) -> float:
        """Estimate memory usage in GB."""
        # Rough estimation based on model size and batch size
        model_params = config.model.z_dim * sum(config.model.encoder_config['hidden_dims'])
        batch_memory = config.data.batch_size * config.model.z_dim * 4  # float32
        
        # Very rough estimate
        total_bytes = (model_params + batch_memory) * 8  # 8 bytes per parameter/activation
        return total_bytes / (1024 ** 3)  # Convert to GB
    
    def _get_available_memory(self) -> float:
        """Get available memory in GB."""
        if torch.cuda.is_available():
            # GPU memory
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            # System RAM (rough estimate)
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
    
    def create_experiment_configs(self, 
                                 base_config: ExperimentConfig,
                                 param_grid: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """
        Create multiple configs for parameter sweeps.
        
        Args:
            base_config: Base configuration
            param_grid: Dictionary of parameter lists
            
        Returns:
            List of experiment configurations
        """
        from itertools import product
        
        configs = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for value_combination in product(*param_values):
            config = deepcopy(base_config)
            
            # Apply parameter combination
            overrides = dict(zip(param_names, value_combination))
            self._apply_overrides(config, overrides)
            
            # Update experiment name
            param_str = '_'.join([f"{k}={v}" for k, v in overrides.items()])
            config.name = f"{base_config.name}_{param_str}"
            
            configs.append(config)
        
        return configs


# Test implementation
if __name__ == "__main__":
    print("=== Testing CSP Configuration Manager ===")
    
    print("\n--- Testing ExperimentConfig creation ---")
    try:
        # Test default config
        config = ExperimentConfig()
        print(f"Default config created: {config.name}")
        print(f"Scenario: {config.data.scenario}")
        print(f"Z-dim: {config.model.z_dim}")
        print(f"Max epochs: {config.training.max_epochs}")
        
        # Test custom config
        custom_config = ExperimentConfig(
            name='test_experiment',
            data=DataConfig(scenario='IY', batch_size=64),
            model=ModelConfig(z_dim=256),
            training=TrainingConfig(max_epochs=50)
        )
        print(f"Custom config created: {custom_config.name}")
        
        print("✓ ExperimentConfig creation test passed")
        
    except Exception as e:
        print(f"✗ ExperimentConfig creation test failed: {e}")
    
    print("\n--- Testing configuration validation ---")
    try:
        # Test valid config
        valid_config = ExperimentConfig()
        
        # Test invalid config
        try:
            invalid_config = ExperimentConfig(
                data=DataConfig(scenario='INVALID')
            )
        except ValueError as e:
            print(f"✓ Caught invalid scenario: {e}")
        
        print("✓ Configuration validation test passed")
        
    except Exception as e:
        print(f"✗ Configuration validation test failed: {e}")
    
    print("\n--- Testing ConfigManager ---")
    try:
        manager = ConfigManager()
        
        # Test presets
        dev_config = manager.get_preset('dev')
        print(f"Dev preset: {dev_config.name}, z_dim={dev_config.model.z_dim}")
        
        research_config = manager.get_preset('research')
        print(f"Research preset: {research_config.name}, z_dim={research_config.model.z_dim}")
        
        # Test config creation with overrides
        custom_config = manager.create_config(
            preset='dev',
            **{'model.z_dim': 32, 'training.max_epochs': 5}
        )
        print(f"Custom config: z_dim={custom_config.model.z_dim}, epochs={custom_config.training.max_epochs}")
        
        # Test validation
        issues = manager.validate_config(custom_config)
        print(f"Validation issues: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        
        print("✓ ConfigManager test passed")
        
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing save/load ---")
    try:
        import tempfile
        
        config = ExperimentConfig(name='test_save_load')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test YAML save/load
            yaml_path = Path(temp_dir) / 'config.yaml'
            config.save(yaml_path)
            loaded_config = ExperimentConfig.load(yaml_path)
            
            assert loaded_config.name == config.name
            assert loaded_config.model.z_dim == config.model.z_dim
            
            # Test JSON save/load
            json_path = Path(temp_dir) / 'config.json'
            config.save(json_path)
            loaded_config_json = ExperimentConfig.load(json_path)
            
            assert loaded_config_json.name == config.name
        
        print("✓ Save/load test passed")
        
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing parameter sweeps ---")
    try:
        manager = ConfigManager()
        base_config = manager.get_preset('dev')
        
        param_grid = {
            'model.z_dim': [32, 64, 128],
            'training.learning_rate': [1e-4, 1e-3, 1e-2]
        }
        
        sweep_configs = manager.create_experiment_configs(base_config, param_grid)
        print(f"Generated {len(sweep_configs)} configurations for sweep")
        
        for i, config in enumerate(sweep_configs[:3]):  # Show first 3
            print(f"  Config {i+1}: z_dim={config.model.z_dim}, lr={config.training.learning_rate}")
        
        print("✓ Parameter sweep test passed")
        
    except Exception as e:
        print(f"✗ Parameter sweep test failed: {e}")
    
    print("\n--- Testing experiment ID generation ---")
    try:
        config1 = ExperimentConfig(name='test')
        config2 = ExperimentConfig(name='test', model=ModelConfig(z_dim=256))
        
        id1 = config1.get_experiment_id()
        id2 = config2.get_experiment_id()
        
        print(f"ID1: {id1}")
        print(f"ID2: {id2}")
        
        assert id1 != id2, "Different configs should have different IDs"
        assert 'test_' in id1, "ID should contain experiment name"
        
        print("✓ Experiment ID generation test passed")
        
    except Exception as e:
        print(f"✗ Experiment ID generation test failed: {e}")
    
    print("\n=== CSP Configuration Manager Test Complete ===")
