import yaml
from pathlib import Path

def load_config(config_file=None):
    """
    Load configuration from a YAML file. Use `.get()` for defaults in the main code.
    If config_file is not provided, defaults to 'config/preprocess_config.yaml'
    in the project root (relative to this script).
    """

    # If no config file is specified, use a default path relative to this script.
    # Example: go up 3 or 4 levels to reach the project root, then into config/preprocess_config.yaml.
    if config_file is None:
        # Adjust parents[...] as necessary based on your directory structure
        default_path = Path(__file__).resolve().parents[3] / "config" / "preprocess_config.yaml"
        config_file = str(default_path)  # convert Path object to string if needed

    try:
        # Load user-defined configuration
        with open(config_file, "r") as file:
            user_config = yaml.safe_load(file)
            if user_config is None:
                user_config = {}
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found. Using only default settings.")
        user_config = {}

    return user_config
