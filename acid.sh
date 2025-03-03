#!/bin/bash

CUDA_DEVICE=0

# Loop through the pruning percentages
for PRUNE_PERCENT in 90 80 70 60 50 40
do
    echo "Running experiment with PRUNE_PERCENT=$PRUNE_PERCENT"
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.main +experiment=acid mode=test \
        wandb.name=acid \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
        checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-28_06-15-19/checkpoints/epoch_1-step_44050.ckpt \
        test.save_image=true \
        wandb.mode=offline \
        test.overlap_range=large \
        model.encoder.prune_percent=$PRUNE_PERCENT \
        test.title=Ours_${PRUNE_PERCENT}_pruning > 3D_Aware_On_2D_Checkpoint_${PRUNE_PERCENT}_prune_Acid.out
done
