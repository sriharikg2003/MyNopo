percent=66
CUDA_VISIBLE_DEVICES=1 python -m src.main +experiment=acid mode=test wandb.name=acid \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-28_06-15-19/checkpoints/epoch_1-step_44050.ckpt \
test.save_image=true wandb.mode=offline test.overlap_range=large \
model.encoder.prune_percent=$percent test.title=3D_with_2D_Ours_$percent \
> 3D_with_2D_Ours_${percent}_ACID.out

percent=55
CUDA_VISIBLE_DEVICES=1 python -m src.main +experiment=acid mode=test wandb.name=acid \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-28_06-15-19/checkpoints/epoch_1-step_44050.ckpt \
test.save_image=true wandb.mode=offline test.overlap_range=large \
model.encoder.prune_percent=$percent test.title=3D_with_2D_Ours_$percent \
> 3D_with_2D_Ours_${percent}_ACID.out

percent=44
CUDA_VISIBLE_DEVICES=1 python -m src.main +experiment=acid mode=test wandb.name=acid \
dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-28_06-15-19/checkpoints/epoch_1-step_44050.ckpt \
test.save_image=true wandb.mode=offline test.overlap_range=large \
model.encoder.prune_percent=$percent test.title=3D_with_2D_Ours_$percent \
> 3D_with_2D_Ours_${percent}_ACID.out