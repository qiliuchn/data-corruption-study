#!/bin/bash

# Create the save directory if it doesn't exist
mkdir -p ./save

# Activate the Conda environment
deep_learning_env_name="sumo"
# Check if the Conda environment exists
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$deep_learning_env_name"

# for macos
export DYLD_LIBRARY_PATH="/opt/anaconda3/envs/sumo/lib:$DYLD_LIBRARY_PATH"
export SUMO_HOME="/opt/homebrew/share/sumo"
export PYTHONPATH="/opt/homebrew/share/sumo/tools:$PYTHONPATH"

# tasks
corruption_modes=("vehicle_miss" "insert_noise" "region_mask")
corruption_ratios=(0.2)
#corruption_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
imputation_mode="none"
#imputation_modes=("none" "context_median" "moving_average" "denosing_autoencoder")
num_episodes=4
#num_episodes=50
#num_episodess=(50 100 150 200 250 300 350 400 450 500)
for corruption_mode in "${corruption_modes[@]}"; do
    for corruption_ratio in "${corruption_ratios[@]}"; do
        # Generate output filenames
        train_log_file="./save/train_output_${corruption_mode}_${corruption_ratio}_${imputation_mode}_${num_episodes}.txt"
        episode_returns_dir="./save/episode_returns_${corruption_mode}_${corruption_ratio}_${imputation_mode}_${num_episodes}.json"
        agent_dir="./save/agent_${corruption_mode}_${corruption_ratio}_${imputation_mode}_${num_episodes}.pth"
        figure_dir="./save/Figure_2_${corruption_mode}_${corruption_ratio}_${imputation_mode}_${num_episodes}.png"
        test_log_file="./save/test_output_${corruption_mode}_${corruption_ratio}_${imputation_mode}_${num_episodes}.txt"

        # Run the Python script and redirect outputs
        echo "Running with corruption_mode=$corruption_mode, corruption_ratio=$corruption_ratio, imputation_mode=$imputation_mode, num_episodes=$num_episodes"
        # train the model
        python change_config.py --run_mode train --corruption_mode "$corruption_mode" --corruption_ratio "$corruption_ratio" --imputation_mode "$imputation_mode" --num_episodes "$num_episodes"
        python train.py > "$train_log_file" 2>&1
        # run test program
        python change_config.py --run_mode test --corruption_mode "$corruption_mode" --corruption_ratio "$corruption_ratio" --imputation_mode "$imputation_mode" --num_episodes "$num_episodes"
        python test_autorun.py > "$test_log_file" 2>&1
        # save agent, figure, and episode return files
        mv agent.pth "$agent_dir"
        mv Figure_2.png "$figure_dir"
        mv episode_returns.json "$episode_returns_dir"
    done
done