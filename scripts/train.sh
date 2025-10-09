#!/bin/bash

# Define arrays for n_shot and a_shot
n_shot=(1 2 4)
a_shot=(1)

# Get command line arguments
gpu=$1
save_dir=$2
fold=$3

# Create the save directory if it doesn't exist
mkdir -p "$save_dir"

# Loop through each combination of n_shot and a_shot
for n in "${n_shot[@]}"; do
  for a in "${a_shot[@]}"; do
    # Generate a random port for torchrun
    port=$((RANDOM % 64512 + 1024))

    echo "Starting training for n_shot=${n}, a_shot=${a} on GPU ${gpu}, saving to ${save_dir}"

    # Run the training command
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$port train.py \
      --data_root /gpfs/work/int/yuexinwang23/Datasets \
      --fold "$fold" \
      --epoch 20 \
      --batch_size 8 \
      --image_size 448 \
      --print_freq 50 \
      --n_shot "$n" \
      --a_shot "$a" \
      --num_learnable_proxies 25 \
      --save_path "$save_dir" \
      | tee "${save_dir}/n_${n}_a_${a}.log" # Redirect output to a log file
  done
done

echo "All training settings completed."
