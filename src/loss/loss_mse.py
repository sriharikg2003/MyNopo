from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

import torch.nn.functional as F
@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:


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

        delta = prediction.color - target_gt
        return self.cfg.weight * (delta**2).mean()
