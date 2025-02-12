import torch
checkpoint = torch.load("/workspace/raid/cdsbad/splat3r_try/NoPoSplat/outputs/exp_re10k/2025-02-11_12-56-09/checkpoints/epoch_0-step_20000.ckpt")
print("Checkpoint keys:", checkpoint.keys())

for name, _ in model.named_parameters():
    print("Model key:", name)
