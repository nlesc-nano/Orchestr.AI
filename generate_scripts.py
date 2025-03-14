import yaml
import re
import argparse

def load_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_neuqip_config_to_temp_yaml(neuqip_config, temp_file_name): # Modified to accept filename as arg
    with open(temp_file_name, 'w') as f:
        yaml.dump(neuqip_config, f)
    return temp_file_name

def resolve_placeholders(config, root_config):
    """Recursively resolves placeholders in a configuration dictionary."""
    if isinstance(config, dict):
        resolved_config = {}
        for key, value in config.items():
            resolved_config[key] = resolve_placeholders(value, root_config)
        return resolved_config
    elif isinstance(config, list):
        return [resolve_placeholders(item, root_config) for item in config]
    elif isinstance(config, str):
        match = re.match(r'\${(\w+)\.(\w+)}', config) # Regex to find ${section.parameter}
        if match:
            section, parameter = match.groups()
            if section in root_config and parameter in root_config[section]:
                return root_config[section][parameter]
    return config # Return original value if not a placeholder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate platform-specific configuration YAML files.")
    parser.add_argument("-p","--platform", choices=['schnet', 'nequip'], required=True, help="Platform to generate configuration for (schnet or nequip)")
    parser.add_argument("-o", "--output", dest="output_file", required=True, help="Output YAML file name (e.g., schnet.yaml or nequip.yaml)")
    args = parser.parse_args()

    platform = args.platform
    output_file_name = args.output_file

    merged_config = load_config("merged_config.yaml")


    if platform == "schnet":
        schnet_settings = merged_config['schnet_config']['settings'] # Access settings from merged_config
        print("Using Schnet Configuration:")

        # 1. Configuration - Resolve placeholders and WRAP under 'settings' BEFORE dumping
        resolved_schnet_settings = resolve_placeholders(schnet_settings, merged_config) # Resolve placeholders!
        schnet_config_with_settings = {'settings': resolved_schnet_settings} # WRAP under 'settings'
        with open(output_file_name, 'w') as f: # Use output_file_name from argument
             yaml.dump(schnet_config_with_settings, f) # Dump the wrapped config

        print(f"Schnet configuration saved to: {output_file_name}") # Modified output for bash script to capture

    elif platform == "nequip":
        neuqip_settings = merged_config['neuqip_config']
        print("\nUsing Neuqip Configuration:")

        # 1. Save Neuqip config to temporary YAML - Resolve placeholders BEFORE dumping

        config_file_name = output_file_name # Use output_file_name from argument as the final name
        resolved_neuqip_settings = resolve_placeholders(neuqip_settings, merged_config) # Resolve placeholders!
        save_neuqip_config_to_temp_yaml(resolved_neuqip_settings, config_file_name) # Save resolved to final output file

        print(f"Neuqip configuration saved to: {config_file_name}") # Modified output for bash script to capture

    else:
        print(f"Error: Unknown platform '{platform}'. Please choose 'schnet' or 'neuqip'.")
        exit()