
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg 
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
import numpy as np

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

    # def forward(
    #     self,
    #     gaussians: Gaussians,
    #     extrinsics: Float[Tensor, "batch view 4 4"],
    #     intrinsics: Float[Tensor, "batch view 3 3"],
    #     near: Float[Tensor, "batch view"],
    #     far: Float[Tensor, "batch view"],
    #     image_shape: tuple[int, int],
    #     depth_mode: DepthRenderingMode | None = None,
    #     cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
    #     cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    #      **kwargs
    # ) -> DecoderOutput:
    #     b, v, _, _ = extrinsics.shape


    #     h , w = image_shape


    #     kwargs = {k: v for k, v in kwargs.items() if k in ["patch_size", "patch_loc", "which_img"]} 
    #     row_start1, row_end1, col_start1, col_end1 , row_start2, row_end2, col_start2, col_end2  = kwargs["patch_loc"] 


        
        
    #     rows = (row_end1[0] - row_start1[0]).item()  
    #     cols = (col_end1[0] - col_start1[0]).item()

    #     total_points = rows * cols
    #     num_to_zero = int(0.8 * total_points)

    #     flat_indices = np.random.choice(total_points, num_to_zero, replace=False)
    #     row_indices, col_indices = np.unravel_index(flat_indices, (rows, cols))

    #     row_start_cpu = row_start1[0].cpu().numpy()  
    #     col_start_cpu = col_start1[0].cpu().numpy()

    #     row_indices_global = row_indices + row_start_cpu
    #     col_indices_global = col_indices + col_start_cpu

        


    #     gaussians_means_reshaped = gaussians.means.view(b, 2, h, w, 3)
    #     gaussians_covariances_reshaped = gaussians.covariances.view(b, 2, h, w, 3, 3)
    #     gaussians_harmonics_reshaped = gaussians.harmonics.view(b, 2, h, w, 3, 25)
    #     gaussians_opacities_reshaped = gaussians.opacities.view(b, 2, h, w)




    #     for i in range(len(row_start1)):
    #         gaussians_means_reshaped[:,:, row_indices_global , col_indices_global    , :]  = 0
    #         gaussians_covariances_reshaped[:,: , row_indices_global , col_indices_global     , :,:]   = 0
    #         gaussians_harmonics_reshaped[ : , : ,    row_indices_global , col_indices_global   ,   : , :]   = 0
    #         gaussians_opacities_reshaped [:,: , row_indices_global , col_indices_global   ]   =0



    #     # for i in range(len(row_start1)):
    #     #     gaussians_means_reshaped[:,:, row_start1[i] : row_end1[i] ,   col_start1[i] : col_end1[i]    , :]  = 0
    #     #     gaussians_covariances_reshaped[:,: , row_start1[i] : row_end1[i] ,   col_start1[i] : col_end1[i]     , :,:]   = 0
    #     #     gaussians_harmonics_reshaped[ : , : ,    row_start1[i] : row_end1[i] ,   col_start1[i] : col_end1[i]   ,   : , :]   = 0
    #     #     gaussians_opacities_reshaped [:,: , row_start1[i] : row_end1[i] ,   col_start1[i] : col_end1[i]   ]   =0

    #         # gaussians_means_reshaped[:,:, row_start2[i] : row_end2[i] ,   col_start2[i] : col_end2[i]    , :]  = 0 
    #         # gaussians_covariances_reshaped[:,:  row_start2[i] : row_end2[i] ,   col_start2[i] : col_end2[i]     , :,:]   = 0 
    #         # gaussians_harmonics_reshaped[ : , : ,    row_start2[i] : row_end2[i] ,   col_start2[i] : col_end2[i]   ,   : , :]   = 0
    #         # gaussians_opacities_reshaped [:,: , row_start2[i] : row_end2[i] ,   col_start2[i] : col_end2[i]   ]   =0





    #     gaussians.means = gaussians_means_reshaped.view(b,2*h*w , 3)
    #     gaussians.covariances = gaussians_covariances_reshaped.view(b,2*h*w , 3,3)
    #     gaussians.harmonics = gaussians_harmonics_reshaped.view(b,2*h*w , 3,-1)
    #     gaussians.opacities = gaussians_opacities_reshaped.view(b,2*h*w)

        
    #     color, depth = render_cuda(
    #         rearrange(extrinsics, "b v i j -> (b v) i j"),
    #         rearrange(intrinsics, "b v i j -> (b v) i j"),
    #         rearrange(near, "b v -> (b v)"),
    #         rearrange(far, "b v -> (b v)"),
    #         image_shape,
    #         repeat(self.background_color, "c -> (b v) c", b=b, v=v),
    #         repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
    #         repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
    #         repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
    #         repeat(gaussians.opacities, "b g -> (b v) g", v=v),
    #         scale_invariant=self.make_scale_invariant,
    #         cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
    #         cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
    #     )
    #     color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

    #     depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v)
    #     return DecoderOutput(color, depth)
    

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


        kwargs = {k: v for k, v in kwargs.items() if k in ["rep"]} 
        rep = kwargs["rep"] 
        
            



        gaussians_means_reshaped = gaussians.means.view(b, 2, h, w, 3)
        gaussians_covariances_reshaped = gaussians.covariances.view(b, 2, h, w, 3, 3)
        gaussians_harmonics_reshaped = gaussians.harmonics.view(b, 2, h, w, 3, 25)
        gaussians_opacities_reshaped = gaussians.opacities.view(b, 2, h, w)



        mask_expanded_1 = rep.unsqueeze(-1)  
        mask_expanded_2 = rep.unsqueeze(-1).unsqueeze(-1)  # For tensors with shape [1, 2, 256, 256, 3, 3] and [1, 2, 256, 256, 3, 25]


        gaussians_means_reshaped = gaussians_means_reshaped * mask_expanded_1  # [1, 2, 256, 256, 3]
        gaussians_covariances_reshaped = gaussians_covariances_reshaped * mask_expanded_2  # [1, 2, 256, 256, 3, 3]
        gaussians_harmonics_reshaped = gaussians_harmonics_reshaped * mask_expanded_2  # [1, 2, 256, 256, 3, 25]
        gaussians_opacities_reshaped = gaussians_opacities_reshaped * rep



        # # Superpoint 

        import torch
        import numpy as np
        from sklearn.cluster import KMeans


        b, _, h, w, _ = gaussians_means_reshaped.shape
        num_points = 2 * h * w  
        means_flat = gaussians_means_reshaped.view(b, num_points, 3) 
        covariances_flat = gaussians_covariances_reshaped.view(b, num_points, 3, 3)  
        harmonics_flat = gaussians_harmonics_reshaped[..., 0].view(b, num_points, 3, 1)  
        opacities_flat = gaussians_opacities_reshaped.view(b, num_points)  
        # 

        data = torch.cat([
            means_flat, 
            harmonics_flat.view(b, num_points, -1),
            
        ], dim=-1)  
        data_np = data.detach().cpu().numpy().reshape(-1, data.shape[-1])  #
        num_clusters = 300  # Adjust as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_np)  # [B*N]

        mask = np.ones(data_np.shape[0], dtype=bool)

        for i in range(num_clusters):
            cluster_indices = np.where(clusters == i)[0] 
            np.random.shuffle(cluster_indices)
            mask[cluster_indices[:len(cluster_indices) // 2]] = 0 

  
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=data.device).view(b, num_points)

        means_flat[~mask_tensor] = 0
        covariances_flat[~mask_tensor] = 0
        harmonics_flat[~mask_tensor] = 0
        opacities_flat[~mask_tensor] = 0


        gaussians.means = means_flat.view(b, 2*h*w, 3)
        gaussians.covariances = covariances_flat.view(b, 2*h*w, 3, 3)
        gaussians.harmonics = harmonics_flat.view(b, 2*h*w, 3, 1) 
        gaussians.opacities = opacities_flat.view(b, 2*h*w)




        # Unstrided
        # gaussians.means = gaussians_means_reshaped.view(b,2*h*w , 3)
        # gaussians.covariances = gaussians_covariances_reshaped.view(b,2*h*w , 3,3)
        # gaussians.harmonics = gaussians_harmonics_reshaped.view(b,2*h*w , 3,-1)
        # gaussians.opacities = gaussians_opacities_reshaped.view(b,2*h*w)


        # Strided
        # stride = 4
        # h_new, w_new = h // stride, w // stride

        # gaussians_means_reshaped = gaussians_means_reshaped[:, :, ::stride, ::stride, :]
        # gaussians_covariances_reshaped = gaussians_covariances_reshaped[:, :, ::stride, ::stride, :, :]
        # gaussians_harmonics_reshaped = gaussians_harmonics_reshaped[:, :, ::stride, ::stride, :, :]
        # gaussians_opacities_reshaped = gaussians_opacities_reshaped[:, :, ::stride, ::stride]

        # gaussians.means = gaussians_means_reshaped.reshape(b, 2 * h_new * w_new, 3)
        # gaussians.covariances = gaussians_covariances_reshaped.reshape(b, 2 * h_new * w_new, 3, 3)
        # gaussians.harmonics = gaussians_harmonics_reshaped.reshape(b, 2 * h_new * w_new, 3, -1)
        # gaussians.opacities = gaussians_opacities_reshaped.reshape(b, 2 * h_new * w_new)



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
        print("****")
        # import torchvision
        # torchvision.utils.save_image(depth[0][1] , f"depth_{stride}.png")
        # torchvision.utils.save_image(color[0][1] , f"color_{stride}.png")
        return DecoderOutput(color, depth)