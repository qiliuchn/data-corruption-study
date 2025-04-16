#!/bin/bash

# Define the values for each argument
corruption_modes=("miss" "noise")
corruption_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
imputation_modes=("none" "unk" "wordvec" "bert")

# Create the save directory if it doesn't exist
mkdir -p ./save

# Activate the "torch" Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"  # Load Conda functions
conda activate torch

# Loop over all combinations of arguments
for corruption_mode in "${corruption_modes[@]}"; do
  for corruption_ratio in "${corruption_ratios[@]}"; do
    for imputation_mode in "${imputation_modes[@]}"; do
      
      # Generate output filenames
      log_file="output_${corruption_mode}_${corruption_ratio}_${imputation_mode}.txt"
      save_dir="./save/logs_${corruption_mode}_${corruption_ratio}_${imputation_mode}"

      # Run the Python script and redirect outputs
      echo "Running with corruption_mode=$corruption_mode, corruption_ratio=$corruption_ratio, imputation_mode=$imputation_mode"
      python finetune.py --corruption_mode "$corruption_mode" --corruption_ratio "$corruption_ratio" --imputation_mode "$imputation_mode" > "$log_file" 2>&1

      # Move the logs directory to the save directory
      if [ -d "./logs" ]; then
        mv ./logs "$save_dir"
      else
        echo "Warning: Logs directory './logs' not found. Skipping move."
      fi
    done
  done
done