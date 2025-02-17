from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.
        near = batch["target"]["near"][..., None, None].log()
        far = batch["target"]["far"][..., None, None].log()
        depth = prediction.depth.minimum(far).maximum(near)
        depth = (depth - near) / (far - near)

        # Compute the difference between neighboring pixels in each direction.
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # If desired, compute a 2nd derivative.
        if self.cfg.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        # If desired, add bilateral filtering.
        if self.cfg.sigma_image is not None:



            target_gt = batch["target"]["image"]
            batch_size = target_gt.shape[0]
            num_targets = target_gt.shape[1]
            channels = target_gt.shape[2]

            device = target_gt.device 


            resized_target_gt = torch.zeros(batch_size, num_targets, channels, 128, 128, device=device)


            for i in range(batch_size):
                for j in range(num_targets):
                    resized_target_gt[i, j] = F.interpolate(target_gt[i, j].unsqueeze(0).to(device), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)


            target_gt = resized_target_gt

            color_dx = reduce(target_gt.diff(dim=-1), "b v c h w -> b v h w", "max")
            color_dy = reduce(target_gt.diff(dim=-2), "b v c h w -> b v h w", "max")
            if self.cfg.use_second_derivative:
                color_dx = color_dx[..., :, 1:].maximum(color_dx[..., :, :-1])
                color_dy = color_dy[..., 1:, :].maximum(color_dy[..., :-1, :])
            depth_dx = depth_dx * torch.exp(-color_dx * self.cfg.sigma_image)
            depth_dy = depth_dy * torch.exp(-color_dy * self.cfg.sigma_image)

        return self.cfg.weight * (depth_dx.abs().mean() + depth_dy.abs().mean())
