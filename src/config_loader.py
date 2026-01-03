"""Configuration loader for SFT training pipeline."""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for SFT training."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute access.
        
        Args:
            name: Configuration key
            
        Returns:
            Configuration value
        """
        # Use object.__getattribute__ to avoid infinite recursion during pickling
        try:
            config_dict = object.__getattribute__(self, '_config')
        except AttributeError:
            raise AttributeError(f"Config object has no attribute '{name}'")
        
        if name in config_dict:
            value = config_dict[name]
            # Recursively convert nested dicts to Config objects
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        value = self._config.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return self._config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['model', 'training', 'evaluation', 'data', 'checkpointing', 'logging']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return Config(config_dict)


def validate_config(config: Config) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model config
    if not config.model.name:
        raise ValueError("Model name cannot be empty")
    
    # Validate training config
    if config.training.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    # Check for batch_size (SFT format) or batch_size_per_gpu (GRPO format)
    batch_size = config.training.get('batch_size')
    batch_size_per_gpu = config.training.get('batch_size_per_gpu')
    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if batch_size_per_gpu is not None and batch_size_per_gpu <= 0:
        raise ValueError("Batch size per GPU must be positive")
    if batch_size is None and batch_size_per_gpu is None:
        raise ValueError("Either batch_size or batch_size_per_gpu must be specified")
    
    # Validate devices (only for SFT configs that have device fields)
    valid_devices = ['cuda:0', 'cuda:1', 'cuda', 'cpu']
    training_device = config.training.get('device')
    if training_device is not None and training_device not in valid_devices:
        raise ValueError(f"Invalid training device: {training_device}")
    
    eval_device = config.evaluation.get('device') if hasattr(config, 'evaluation') else None
    if eval_device is not None and eval_device not in valid_devices:
        raise ValueError(f"Invalid evaluation device: {eval_device}")
    
    # Validate checkpointing (only if queue_maxsize is specified)
    if hasattr(config, 'checkpointing'):
        queue_maxsize = config.checkpointing.get('queue_maxsize')
        if queue_maxsize is not None and queue_maxsize <= 0:
            raise ValueError("Queue maxsize must be positive")
    
    print("✓ Configuration validated successfully")
