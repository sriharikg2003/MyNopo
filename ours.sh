CUDA_VISIBLE_DEVICES=0  python -m src.main +experiment=acid mode=test wandb.name=acid dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-28_06-15-19/checkpoints/epoch_1-step_44050.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=60 test.overlap_range=large test.title=Ours_60_pruning > 2D_Stero_Ours_60_pruning_Acid.out

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-28_06-14-55/checkpoints/epoch_0-step_47500.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=60 test.overlap_range=large test.title=Ours_60_pruning > 2D_Stero_Ours_60_pruning_RE10.out





