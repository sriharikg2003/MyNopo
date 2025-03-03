#!/bin/bash

CUDA_DEVICE=1

# Loop through the pruning percentages
for PRUNE_PERCENT in 90 80 70 60 50 40
do
    echo "Running RE10k experiment with PRUNE_PERCENT=$PRUNE_PERCENT"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.main +experiment=re10k mode=test \
        wandb.name=re10k \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
        checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-28_06-14-55/checkpoints/epoch_0-step_47500.ckpt \
        test.save_image=true \
        model.encoder.prune_percent=$PRUNE_PERCENT \
        wandb.mode=offline \
        test.overlap_range=large \
        test.title=Ours_${PRUNE_PERCENT}_pruning > 3D_Aware_On_2D_Checkpoint_${PRUNE_PERCENT}_prune_Re10.out
done
