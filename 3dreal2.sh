percent=66
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-28_06-14-55/checkpoints/epoch_0-step_47500.ckpt \
test.save_image=true wandb.mode=offline test.overlap_range=large \
model.encoder.prune_percent=$percent test.title=3D_with_2D_Ours_$percent \
> 3D_with_2D_Ours_${percent}_REAL.out

percent=55
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-28_06-14-55/checkpoints/epoch_0-step_47500.ckpt \
test.save_image=true wandb.mode=offline test.overlap_range=large \
model.encoder.prune_percent=$percent test.title=3D_with_2D_Ours_$percent \
> 3D_with_2D_Ours_${percent}_REAL.out

percent=44
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-28_06-14-55/checkpoints/epoch_0-step_47500.ckpt \
test.save_image=true wandb.mode=offline test.overlap_range=large \
model.encoder.prune_percent=$percent test.title=3D_with_2D_Ours_$percent \
> 3D_with_2D_Ours_${percent}_REAL.out
