CUDA_VISIBLE_DEVICES=0  python -m src.main +experiment=acid mode=test wandb.name=acid dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-26_11-20-51/checkpoints/epoch_0-step_16410.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=65 test.overlap_range=large test.title=65_pruning > 2D_Stero_65_prune_Acid.out


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-26_11-20-56/checkpoints/epoch_0-step_25880.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=65 test.overlap_range=large test.title=65_pruning > 2D_Stero_65_prune_RE10.out



CUDA_VISIBLE_DEVICES=0  python -m src.main +experiment=acid mode=test wandb.name=acid dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-26_11-20-51/checkpoints/epoch_0-step_16410.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=85 test.overlap_range=large test.title=85_pruning > 2D_Stero_85_prune_Acid.out


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-26_11-20-56/checkpoints/epoch_0-step_25880.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=85 test.overlap_range=large test.title=85_pruning > 2D_Stero_85_prune_RE10.out



CUDA_VISIBLE_DEVICES=0  python -m src.main +experiment=acid mode=test wandb.name=acid dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-26_11-20-51/checkpoints/epoch_0-step_16410.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=95 test.overlap_range=large test.title=95_pruning > 2D_Stero_95_prune_Acid.out


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-26_11-20-56/checkpoints/epoch_0-step_25880.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=95 test.overlap_range=large test.title=95_pruning > 2D_Stero_95_prune_RE10.out


CUDA_VISIBLE_DEVICES=0  python -m src.main +experiment=acid mode=test wandb.name=acid dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_acid/2025-02-26_11-20-51/checkpoints/epoch_0-step_16410.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=0 test.overlap_range=large test.title=0_pruning > 2D_Stero_0_prune_Acid.out


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k mode=test wandb.name=re10k dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-26_11-20-56/checkpoints/epoch_0-step_25880.ckpt test.save_image=true wandb.mode=offline dataset.re10k.prune_percent=0 test.overlap_range=large test.title=0_pruning > 2D_Stero_0_prune_RE10.out