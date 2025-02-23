
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import time
from ...dataset import DatasetCfg 
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
import numpy as np

import numpy as np
import torch
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from plyfile import PlyData, PlyElement
import copy
import torch
import numpy as np
import einops
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
         **kwargs
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape


        h , w = image_shape

        original_gaussians = Gaussians(
            means=gaussians.means.clone().detach(),
            covariances=gaussians.covariances.clone().detach(),
            harmonics=gaussians.harmonics.clone().detach(),
            opacities=gaussians.opacities.clone().detach()
        )

        original = kwargs["original"]
        kwargs = {k: v for k, v in kwargs.items() if k in ["rep"]} 
        rep = kwargs["rep"] 

        if not original:
        
            # gaussians_means_reshaped = gaussians.means.view(b, 2, h, w, 3)
            # gaussians_covariances_reshaped = gaussians.covariances.view(b, 2, h, w, 3, 3)
            # gaussians_harmonics_reshaped = gaussians.harmonics.view(b, 2, h, w, 3, 25)
            # gaussians_opacities_reshaped = gaussians.opacities.view(b, 2, h, w)



            # mask_expanded_1 = rep.unsqueeze(-1)  
            # mask_expanded_2 = rep.unsqueeze(-1).unsqueeze(-1)  # For tensors with shape [1, 2, 256, 256, 3, 3] and [1, 2, 256, 256, 3, 25]


            # gaussians_means_reshaped = gaussians_means_reshaped * mask_expanded_1  # [1, 2, 256, 256, 3]
            # gaussians_covariances_reshaped = gaussians_covariances_reshaped * mask_expanded_2  # [1, 2, 256, 256, 3, 3]
            # gaussians_harmonics_reshaped = gaussians_harmonics_reshaped * mask_expanded_2  # [1, 2, 256, 256, 3, 25]
            # gaussians_opacities_reshaped = gaussians_opacities_reshaped * rep


            # gaussians.means = gaussians_means_reshaped.view(b,2*h*w , 3)
            # gaussians.covariances = gaussians_covariances_reshaped.view(b,2*h*w , 3,3)
            # gaussians.harmonics = gaussians_harmonics_reshaped.view(b,2*h*w , 3,-1)
            # gaussians.opacities = gaussians_opacities_reshaped.view(b,2*h*w)


            # breakpoint()
            # gaussians.means = gaussians.means[ rep.reshape(b,-1) ].unsqueeze(0)
            # gaussians.covariances = gaussians.covariances[ rep.reshape(b,-1) ].unsqueeze(0)
            # gaussians.harmonics = gaussians.harmonics[ rep.reshape(b,-1) ].unsqueeze(0)
            # gaussians.opacities = gaussians.opacities[ rep.reshape(b,-1) ].unsqueeze(0)

            pass
        breakpoint()
        color, depth = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=self.make_scale_invariant,
            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
        )


    
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v)




        # import torchvision
        # torchvision.utils.save_image(depth[0][1] , f"depth_{stride}.png")
        # torchvision.utils.save_image(color[0][1] , f"color_{stride}.png")
        return DecoderOutput(color, depth , original_gaussians)