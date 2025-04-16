import argparse
import yaml

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config, config_path):
    """Save the modified configuration back to the YAML file."""
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def update_config(config, run_mode, corruption_mode, corruption_ratio, imputation_mode, num_episodes):
    """Update the configuration dictionary with new values."""
    config['run_mode'] = run_mode
    config['corruption_mode'] = corruption_mode
    config['corruption_ratio'] = float(corruption_ratio)
    config['imputation_mode'] = imputation_mode
    config['num_episodes'] = num_episodes
    return config

def update_config_function(config_path, run_mode, corruption_mode, corruption_ratio, imputation_mode, num_episodes):
    config = load_config(config_path)
    config['run_mode'] = run_mode
    config['corruption_mode'] = corruption_mode
    config['corruption_ratio'] = float(corruption_ratio)
    config['imputation_mode'] = imputation_mode
    config['num_episodes'] = num_episodes
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify config.yaml with new hyperparameters.")
    parser.add_argument('--config_path', type=str, default='config.yaml', 
                        help="Path to the configuration file.")
    parser.add_argument('--run_mode', type=str, default='train', choices=["train", "test"],
                        help="Run mode train or test.")
    parser.add_argument('--corruption_mode', type=str, choices=["none", "vehicle_miss", "insert_noise", "region_mask"], required=True,
                        help="Mode of corruption to set in the configuration.")
    parser.add_argument('--corruption_ratio', type=float, required=True,
                        help="Ratio of corruption to set in the configuration.")
    parser.add_argument('--imputation_mode', type=str, choices=["none", "context_fill", "moving_average", "denosing_autoencoder"], required=True,
                        help="Mode of imputation to set in the configuration.")
    parser.add_argument('--num_episodes', type=int, required=True,
                        help="Number of episodes to run the agent for.")
    
    args = parser.parse_args()

    # Load the configuration file
    config = load_config(args.config_path)

    # Update configuration
    updated_config = update_config(config, args.run_mode, args.corruption_mode, args.corruption_ratio, args.imputation_mode, args.num_episodes)

    # Save the updated configuration
    save_config(updated_config, args.config_path)

    print(f"Configuration updated and saved to {args.config_path}")