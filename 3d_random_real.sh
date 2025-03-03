#!/bin/bash

# Accept arguments
PERCENT=$1       # Second argument: Percentage for prune_percent
CUDA_DEVICE=$2   # First argument: CUDA device ID

# Ensure both arguments are provided
if [ -z "$CUDA_DEVICE" ] || [ -z "$PERCENT" ]; then
  echo "Usage: $0 <CUDA_DEVICE> <PERCENT>"
  exit 1
fi

# Set CUDA environment variable and execute the command
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.main +experiment=re10k mode=test wandb.name=re10k \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
  checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-03-03_05-46-58/checkpoints/epoch_0-step_1410.ckpt \
  test.save_image=true wandb.mode=offline test.overlap_range=large \
  model.encoder.prune_percent=$PERCENT test.title=3D_with_Random_Pick_$PERCENT \
  > 3D_with_Random_Pick_${PERCENT}_REAL.out
