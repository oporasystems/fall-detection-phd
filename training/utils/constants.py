import os
import yaml


def _deep_merge(base, override):
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config():
    """Load config from config.yaml, with optional overrides from config.local.yaml."""
    # Find project root (where config.yaml lives)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    base_config_path = os.path.join(project_root, 'config.yaml')
    local_config_path = os.path.join(project_root, 'config.local.yaml')

    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge local overrides if they exist
    if os.path.exists(local_config_path):
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)
            if local_config:
                config = _deep_merge(config, local_config)

    return config


# Load configuration
_config = _load_config()

# Dataset paths (backward compatible exports)
fall_folder = _config['dataset']['fall_folder']
non_fall_folder = _config['dataset']['non_fall_folder']

# Model constants
max_sequence_length = _config['model']['max_sequence_length']
input_dim = _config['model']['input_dim']
num_classes = _config['model']['num_classes']
csv_columns = _config['model']['csv_columns']

# Full config available if needed
config = _config
