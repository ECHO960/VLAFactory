"""Default configuration for VLA training."""

DEFAULT_VLA_CONFIG = {
    # Model configuration
    "model": {
        "action_dim": 7,
        "hidden_dim": 768,
        "model_name_or_path": "meta-llama/Llama-2-7b-hf",
    },
    
    # Dataset configuration
    "data": {
        "data_path": "data/vla_data.json",
        "max_length": 512,
        "train_split": 0.9,
    },
    
    # Training configuration
    "training": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "warmup_steps": 100,
        "logging_steps": 10,
        "eval_epochs": 1,
        "save_steps": 500,
        "output_dir": "./output",
    },
    
    # LlamaFactory integration
    "llamafactory": {
        "stage": "sft",  # Supervised fine-tuning
        "finetuning_type": "lora",  # Can be "full", "lora", or "freeze"
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
}


def get_default_config():
    """Get the default VLA configuration."""
    return DEFAULT_VLA_CONFIG.copy()


def load_config(config_path: str):
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    import json
    import os
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            try:
                import yaml
                config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML configs")
        else:
            raise ValueError("Config file must be JSON or YAML")
    
    # Merge with defaults
    default_config = get_default_config()
    
    # Deep merge
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
        elif isinstance(default_config[key], dict):
            for subkey in default_config[key]:
                if subkey not in config[key]:
                    config[key][subkey] = default_config[key][subkey]
    
    return config


def save_config(config: dict, save_path: str):
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
