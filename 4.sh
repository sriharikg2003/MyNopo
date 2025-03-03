#!/bin/bash
CUDA_DEVICE=1

# Run ACID experiment
PRUNE_PERCENT=50
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.main +experiment=acid mode=test \
    wandb.name=acid \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
    checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-28_12-18-24/checkpoints/epoch_0-step_1160.ckpt \
    test.save_image=true \
    wandb.mode=offline \
    test.overlap_range=large \
    model.encoder.prune_percent=$PRUNE_PERCENT \
    test.title=Ours_${PRUNE_PERCENT}_pruning > 3D_Aware_${PRUNE_PERCENT}_prune_Acid.out

PRUNE_PERCENT=40
# Run RE10k experiment
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.main +experiment=re10k mode=test \
    wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-27_08-41-30/checkpoints/epoch_0-step_2620.ckpt \
    test.save_image=true \
    model.encoder.prune_percent=$PRUNE_PERCENT \
    wandb.mode=offline \
    test.overlap_range=large \
    test.title=Ours_${PRUNE_PERCENT}_pruning > 3D_Aware_${PRUNE_PERCENT}_prune_Re10.out


PRUNE_PERCENT=40
# Run ACID experiment
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.main +experiment=acid mode=test \
    wandb.name=acid \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
    checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-28_12-18-24/checkpoints/epoch_0-step_1160.ckpt \
    test.save_image=true \
    wandb.mode=offline \
    test.overlap_range=large \
    model.encoder.prune_percent=$PRUNE_PERCENT \
    test.title=Ours_${PRUNE_PERCENT}_pruning > 3D_Aware_${PRUNE_PERCENT}_prune_Acid.out