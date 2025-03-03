#!/bin/bash

# Accept arguments
PERCENT=$1       # Second argument: Percentage for prune_percent
CUDA_DEVICE=$2   # First argument: CUDA device ID

# Ensure both arguments are provided
if [ -z "$CUDA_DEVICE" ] || [ -z "$PERCENT" ]; then
  echo "Usage: $0 <CUDA_DEVICE> <PERCENT>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.main +experiment=acid mode=test \
    wandb.name=acid \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
    checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-03-03_05-49-20/checkpoints/epoch_0-step_1640.ckpt \
    test.save_image=true \
    wandb.mode=offline \
    test.overlap_range=large \
    model.encoder.prune_percent=$PERCENT \
    test.title=3D_with_Random_Pick_${PERCENT}_pruning > 3D_with_Random_Pick_${PERCENT}_prune_Acid.out