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
        if name in self._config:
            value = self._config[name]
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
    if config.training.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    # Validate devices
    valid_devices = ['cuda:0', 'cuda:1', 'cuda', 'cpu']
    if config.training.device not in valid_devices:
        raise ValueError(f"Invalid training device: {config.training.device}")
    if config.evaluation.device not in valid_devices:
        raise ValueError(f"Invalid evaluation device: {config.evaluation.device}")
    
    # Validate checkpointing
    if config.checkpointing.queue_maxsize <= 0:
        raise ValueError("Queue maxsize must be positive")
    
    print("✓ Configuration validated successfully")
