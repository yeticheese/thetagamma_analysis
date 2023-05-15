import os
import yaml


def load_config(config_path):
    """Load configuration settings from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Load the configuration settings based on the environment
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
config_path = os.path.join(os.path.dirname(__file__), "src", "configs", f"{ENVIRONMENT}.yaml")
config = load_config(config_path)
