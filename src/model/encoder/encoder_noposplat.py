from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

import torch
import numpy as np
import pywt
import cv2
import random
import matplotlib.pyplot as plt
inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatCfg:
    name: Literal["noposplat", "noposplat_multi"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplat(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type

        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_gs_params_head(self, cfg, head_type):


        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view2 3DGS

        elif head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)



# ORIGINAL FORWARD
    # def forward(
    #     self,
    #     context: dict,
    #     global_step: int = 0,
    #     visualization_dump: Optional[dict] = None,
    # ) -> Gaussians:
    #     device = context["image"].device
    #     b, v, _, h, w = context["image"].shape

    #     # Encode the context images.

    #     # @MODIFIED Remove : 

        

        


    #     with torch.cuda.amp.autocast(enabled=False):


    #         # @MASKED
    #         # for the 3DGS heads
    #         # dec1, dec2,dec1__, dec2__, shape1, shape2, view1, view2,  view1__, view2__  = self.backbone(context, return_views=True)
    #         # res1 = self._downstream_head(1, [tok.float() for tok in dec1]    + [tok.float() for tok in dec1__]  , shape1)
    #         # res2 = self._downstream_head(2, [tok.float() for tok in dec2]  + [tok.float() for tok in dec2__] , shape2)
    #         # if self.gs_params_head_type == 'linear':
    #         #     GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
    #         #     GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
    #         # elif self.gs_params_head_type == 'dpt':
    #         #     GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
    #         #     GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #         #     GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
    #         #     GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
    #         # elif self.gs_params_head_type == 'dpt_gs':
    #         #     GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1]    + [tok.float() for tok in dec1__] , res1['pts3d'].permute(0, 3, 1, 2), (view1['img'][:, :3] , view1__['img'][:, :3]), shape1[0].cpu().tolist())
    #         #     GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #         #     GS_res2 = self.gaussian_param_head2(  [tok.float() for tok in dec2]    + [tok.float() for tok in dec2__] , res2['pts3d'].permute(0, 3, 1, 2), (view2['img'][:, :3], view2__['img'][:, :3]), shape2[0].cpu().tolist())
    #         #     GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

    #         # @UNMASKED
    #         dec1, dec2, shape1, shape2, view1, view2   = self.backbone(context, return_views=True)
    #         res1 = self._downstream_head(1, [tok.float() for tok in dec1]    , shape1)
    #         res2 = self._downstream_head(2, [tok.float() for tok in dec2]   , shape2)
    #         if self.gs_params_head_type == 'linear':
    #             GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
    #             GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
    #         elif self.gs_params_head_type == 'dpt':
    #             GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
    #             GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #             GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
    #             GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
    #         elif self.gs_params_head_type == 'dpt_gs':
    #             GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1]   , res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3] , shape1[0].cpu().tolist())
    #             GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #             GS_res2 = self.gaussian_param_head2(  [tok.float() for tok in dec2]    , res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
    #             GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

    #     pts3d1 = res1['pts3d']
    #     pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
    #     pts3d2 = res2['pts3d']
    #     pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
    #     pts_all = torch.stack((pts3d1, pts3d2), dim=1)
    #     pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces



    #     depths = pts_all[..., -1].unsqueeze(-1)

    #     gaussians = torch.stack([GS_res1, GS_res2], dim=1)
    #     gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
    #     densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

    #     # Convert the features and depths into Gaussians.
    #     if self.pose_free:
    #         gaussians = self.gaussian_adapter.forward(
    #             pts_all.unsqueeze(-2),
    #             depths,
    #             self.map_pdf_to_opacity(densities, global_step),
    #             rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
    #         )
    #     else:
    #         xy_ray, _ = sample_image_grid((h, w), device)
    #         xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
    #         xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

    #         gaussians = self.gaussian_adapter.forward(
    #             rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
    #             rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
    #             rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
    #             depths,
    #             self.map_pdf_to_opacity(densities, global_step),
    #             rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
    #             (h, w),
    #         )

    #     # Dump visualizations if needed.
    #     if visualization_dump is not None:
    #         visualization_dump["depth"] = rearrange(
    #             depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
    #         )
    #         visualization_dump["scales"] = rearrange(
    #             gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
    #         )
    #         visualization_dump["rotations"] = rearrange(
    #             gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
    #         )
    #         visualization_dump["means"] = rearrange(
    #             gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
    #         )
    #         visualization_dump['opacities'] = rearrange(
    #             gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
    #         )

    #     return Gaussians(
    #         rearrange(
    #             gaussians.means,
    #             "b v r srf spp xyz -> b (v r srf spp) xyz",
    #         ),
    #         rearrange(
    #             gaussians.covariances,
    #             "b v r srf spp i j -> b (v r srf spp) i j",
    #         ),
    #         rearrange(
    #             gaussians.harmonics,
    #             "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
    #         ),
    #         rearrange(
    #             gaussians.opacities,
    #             "b v r srf spp -> b (v r srf spp)",
    #         ),
    #     )




    # Cluster Badri , 3D aware pixel 
    # def forward(
    #     self,
    #     context: dict,
    #     global_step: int = 0,
    #     visualization_dump: Optional[dict] = None,
    # ) -> Gaussians:
    #     device = context["image"].device
    #     b, v, _, h, w = context["image"].shape

    #     # Encode the context images.

    #     # @MODIFIED Remove : 

        

    #     print("CAME UP")
        


    #     with torch.cuda.amp.autocast(enabled=False):


    #         # @UNMASKED
    #         dec1, dec2, shape1, shape2, view1, view2   = self.backbone(context, return_views=True)
    #         res1 = self._downstream_head(1, [tok.float() for tok in dec1]    , shape1)
    #         res2 = self._downstream_head(2, [tok.float() for tok in dec2]   , shape2)
    #         if self.gs_params_head_type == 'linear':
    #             GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
    #             GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
    #         elif self.gs_params_head_type == 'dpt':
    #             GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
    #             GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #             GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
    #             GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
    #         elif self.gs_params_head_type == 'dpt_gs':
    #             GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1]   , res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3] , shape1[0].cpu().tolist())
    #             GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #             GS_res2 = self.gaussian_param_head2(  [tok.float() for tok in dec2]    , res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
    #             GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

    #     pts3d1 = res1['pts3d']
    #     pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
    #     pts3d2 = res2['pts3d']
    #     pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
    #     pts_all = torch.stack((pts3d1, pts3d2), dim=1)
    #     pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        

    #     #@BADRI START

    #     # print(pts1_append.shape, pts2_append.shape)
    #     # print(torch.cat((pts1_append, pts2_append), dim=1).shape)
    #     from sklearn.cluster import KMeans
        
    #     #Apply k means in the 3D space
        


    #     for b in range(context['image'].size(0)):  # Iterate over batch
    #           # Debugging breakpoint

    #         # Prepare the 3D point data
    #         pts1_append = torch.cat((res1['pts3d'][b], context['image'][b, 0].permute(1, 2, 0)), dim=-1)
    #         pts2_append = torch.cat((res2['pts3d'][b], context['image'][b, 1].permute(1, 2, 0)), dim=-1)

    #         # Concatenate the tensors for clustering
    #         tensor_data = torch.cat((pts1_append, pts2_append), dim=0).reshape(-1, 6)  # Use dim=0 instead of dim=1

    #         # Ensure tensor is detached before converting to numpy
    #         if tensor_data.requires_grad:
    #             tensor_data = tensor_data.detach()

    #         # Apply K-Means clustering
    #         kmeans = KMeans(n_clusters=300, random_state=0).fit(tensor_data.cpu().numpy())

    #         # Convert results back to tensors
    #         cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(tensor_data.device)
    #         cluster_labels = torch.from_numpy(kmeans.labels_).to(tensor_data.device)

    #           # Debugging breakpoint

    #         # Correct slicing for cluster labels
    #         num_pts1 = pts1_append.shape[0] * pts1_append.shape[1]  # Total points in first image
    #         pts1_cluster_label = cluster_labels[:num_pts1].reshape(pts1_append.shape[0], pts1_append.shape[1])
    #         pts2_cluster_label = cluster_labels[num_pts1:].reshape(pts2_append.shape[0], pts2_append.shape[1])


    #         # BADRI END
            
    #         import matplotlib.pyplot as plt

    #         import numpy as np
    #         import random
    #         super_pixel_coordinates_1 = {i: [] for i in range(pts1_cluster_label.min(), pts1_cluster_label.max() + 1)}
            
    #         for i in range(pts1_cluster_label.shape[0]):
    #             for j in range(pts1_cluster_label.shape[1]):
    #                 super_pixel_coordinates_1[pts1_cluster_label[i, j].cpu().numpy().item()].append((i, j))
    #         num_superpixels = 50
    #         selected_superpixels = random.sample(list(super_pixel_coordinates_1.keys()), num_superpixels)
    #         percentage = 10
    #         representation_gaussians_1 = []
    #         for sp in selected_superpixels:
    #             num_pixels = int(len(super_pixel_coordinates_1[sp]) * percentage / 100)
    #             representation_gaussians_1.extend(random.sample(super_pixel_coordinates_1[sp], num_pixels))
        
    #         mask = np.ones((  context['image'].shape[-2]  , context['image'].shape[-1]), dtype=bool)  
    #         for sp in selected_superpixels:
    #             for x, y in super_pixel_coordinates_1[sp]:
    #                 mask[x, y] = False
    #                 context['image'][b,0,:,x,y] = -1
        
    #         for x, y in representation_gaussians_1:
    #             mask[x, y] = True
        

    #         import torchvision
            
    #         # torchvision.utils.save_image(context['image'][b, 0] , "del.png")
    #         # plt.imshow(mask, cmap="gray")


    #         # plt.savefig(f"mask_{1}.png", bbox_inches='tight')
    #         # plt.close()
        
    #         mask_tensor = torch.tensor(mask, dtype=torch.bool, device=context['rep'][b][0].device)

    #         # Assign to context['rep'][0][1]
    #         context['rep'][b][0] = mask_tensor

        
    #         super_pixel_coordinates_2 = {i: [] for i in range(pts2_cluster_label.min(), pts2_cluster_label.max() + 1)}
    #         for i in range(pts2_cluster_label.shape[0]):
    #             for j in range(pts2_cluster_label.shape[1]):
    #                 super_pixel_coordinates_2[pts2_cluster_label[i, j].cpu().numpy().item()].append((i, j))
    #         representation_gaussians_2 = []
    #         mask = np.ones((   context['image'].shape[-2]  , context['image'].shape[-1]   ), dtype=bool)  
    #         for sp in selected_superpixels:
    #             if sp in super_pixel_coordinates_2:
    #                 num_pixels = int(len(super_pixel_coordinates_2[sp]) * percentage / 100)
    #                 representation_gaussians_2.extend(random.sample(super_pixel_coordinates_2[sp], num_pixels))
                
    #             else:
    #                 print(sp , "******")
    #         for sp in selected_superpixels:
    #             if sp in super_pixel_coordinates_2:
    #                 for x, y in super_pixel_coordinates_2[sp]:
    #                     mask[x, y] = False
    #                     context['image'][b,1,:,x,y] = -1

    #         for x, y in representation_gaussians_2:
    #             mask[x, y] = True

    #         # import torchvision
    #         # torchvision.utils.save_image(context['image'][b, 1] , "del1.png")
    #         # plt.imshow(mask, cmap="gray")


    #         # plt.savefig(f"mask_{2}.png", bbox_inches='tight')
    #         # plt.close()
            


    #         mask_tensor = torch.tensor(mask, dtype=torch.bool, device=context['rep'][b][1].device)

    #         # Assign to context['rep'][0][1]
    #         context['rep'][b][1] = mask_tensor



    #     # END of Masking and all

       


    #     with torch.cuda.amp.autocast(enabled=False):


    #         # @UNMASKED
    #         print("CAME DOWN")
            
    #         dec1, dec2, shape1, shape2, view1, view2   = self.backbone(context, return_views=True)
    #         res1 = self._downstream_head(1, [tok.float() for tok in dec1]    , shape1)
    #         res2 = self._downstream_head(2, [tok.float() for tok in dec2]   , shape2)
    #         if self.gs_params_head_type == 'linear':
    #             GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
    #             GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
    #         elif self.gs_params_head_type == 'dpt':
    #             GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
    #             GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #             GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
    #             GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
    #         elif self.gs_params_head_type == 'dpt_gs':
    #             GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1]   , res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3] , shape1[0].cpu().tolist())
    #             GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
    #             GS_res2 = self.gaussian_param_head2(  [tok.float() for tok in dec2]    , res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
    #             GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

    #     pts3d1 = res1['pts3d']
    #     pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
    #     pts3d2 = res2['pts3d']
    #     pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
    #     pts_all = torch.stack((pts3d1, pts3d2), dim=1)
    #     pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces


    #     depths = pts_all[..., -1].unsqueeze(-1)

    #     gaussians = torch.stack([GS_res1, GS_res2], dim=1)
    #     gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
    #     densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

    #     # Convert the features and depths into Gaussians.
    #     if self.pose_free:
    #         gaussians = self.gaussian_adapter.forward(
    #             pts_all.unsqueeze(-2),
    #             depths,
    #             self.map_pdf_to_opacity(densities, global_step),
    #             rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
    #         )
    #     else:
    #         xy_ray, _ = sample_image_grid((h, w), device)
    #         xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
    #         xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

    #         gaussians = self.gaussian_adapter.forward(
    #             rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
    #             rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
    #             rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
    #             depths,
    #             self.map_pdf_to_opacity(densities, global_step),
    #             rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
    #             (h, w),
    #         )

    #     # Dump visualizations if needed.
    #     if visualization_dump is not None:
    #         visualization_dump["depth"] = rearrange(
    #             depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
    #         )
    #         visualization_dump["scales"] = rearrange(
    #             gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
    #         )
    #         visualization_dump["rotations"] = rearrange(
    #             gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
    #         )
    #         visualization_dump["means"] = rearrange(
    #             gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
    #         )
    #         visualization_dump['opacities'] = rearrange(
    #             gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
    #         )

    #     return Gaussians(
    #         rearrange(
    #             gaussians.means,
    #             "b v r srf spp xyz -> b (v r srf spp) xyz",
    #         ),
    #         rearrange(
    #             gaussians.covariances,
    #             "b v r srf spp i j -> b (v r srf spp) i j",
    #         ),
    #         rearrange(
    #             gaussians.harmonics,
    #             "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
    #         ),
    #         rearrange(
    #             gaussians.opacities,
    #             "b v r srf spp -> b (v r srf spp)",
    #         ),
    #     )



    """
    Wavelet + 3D cluster + Superpixel

    """


    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape




        def select_superpixels(img, sp_cord, segments_slic, wavelet='haar', level=1, percentage=10):
                # Ensure image is on CPU and convert to numpy
            img = img.cpu().numpy()
            
            original_img = img

            if isinstance(segments_slic, torch.Tensor):
                segments_slic = segments_slic.cpu().numpy()
            
            sp_wave_values = {int(i): [] for i in sp_cord.keys()}
            
            coeffs = pywt.wavedec2(original_img[:,:,0], wavelet, level=level)
            _, (x, y, z) = coeffs  # Extract detail coefficients
            what = (x/(np.mean(x)))  +  y/(np.mean(y))  +  z / (np.mean(z)) 
            resized_z = cv2.resize(what, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            for i in range(segments_slic.shape[0]):
                for j in range(segments_slic.shape[1]):
                    key = int(segments_slic[i, j])  # Ensure integer key
                    if key in sp_wave_values:
                        sp_wave_values[key].append(abs(resized_z[i, j]))
                    else:
                        print(f"Warning: Key {key} not found in sp_wave_values")
            mean_wavelet_values = np.array([np.nanmean(sp_wave_values[k]) if np.any(~np.isnan(sp_wave_values[k])) else 0 for k in sp_wave_values.keys()])

            
            
            threshold = np.percentile(mean_wavelet_values, 40)

            selected_superpixels = np.array(list(sp_cord.keys()))[mean_wavelet_values < threshold]

            return selected_superpixels

        # with torch.cuda.amp.autocast(enabled=False):


        #     # @UNMASKED
        #     dec1, dec2, shape1, shape2, view1, view2   = self.backbone(context, return_views=True)
        #     res1 = self._downstream_head(1, [tok.float() for tok in dec1]    , shape1)
        #     res2 = self._downstream_head(2, [tok.float() for tok in dec2]   , shape2)
        #     if self.gs_params_head_type == 'linear':
        #         GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
        #         GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
        #     elif self.gs_params_head_type == 'dpt':
        #         GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
        #         GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
        #         GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
        #         GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
        #     elif self.gs_params_head_type == 'dpt_gs':
        #         GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1]   , res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3] , shape1[0].cpu().tolist())
        #         GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
        #         GS_res2 = self.gaussian_param_head2(  [tok.float() for tok in dec2]    , res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
        #         GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

        # pts3d1 = res1['pts3d']
        # pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        # pts3d2 = res2['pts3d']
        # pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        # pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        # pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        



        # from sklearn.cluster import KMeans
    
        # selected_superpixels_list = []


        # for b in range(context['image'].size(0)):  # Iterate over batch

        #     pts1_append = torch.cat((res1['pts3d'][b], context['image'][b, 0].permute(1, 2, 0)), dim=-1)
        #     pts2_append = torch.cat((res2['pts3d'][b], context['image'][b, 1].permute(1, 2, 0)), dim=-1)

        #     tensor_data = torch.cat((pts1_append, pts2_append), dim=0).reshape(-1, 6)  # Use dim=0 instead of dim=1

        #     if tensor_data.requires_grad:
        #         tensor_data = tensor_data.detach()

        #     kmeans = KMeans(n_clusters=300, random_state=0).fit(tensor_data.cpu().numpy())

        #     # Convert results back to tensors
        #     cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(tensor_data.device)
        #     cluster_labels = torch.from_numpy(kmeans.labels_).to(tensor_data.device)


        #     num_pts1 = pts1_append.shape[0] * pts1_append.shape[1]  # Total points in first image
        #     pts1_cluster_label = cluster_labels[:num_pts1].reshape(pts1_append.shape[0], pts1_append.shape[1])
        #     pts2_cluster_label = cluster_labels[num_pts1:].reshape(pts2_append.shape[0], pts2_append.shape[1])


        #     # BADRI END
            
        #     import matplotlib.pyplot as plt

        #     import numpy as np
        #     import random
        #     super_pixel_coordinates_1 = {i: [] for i in range(pts1_cluster_label.min(), pts1_cluster_label.max() + 1)}
            
        #     for i in range(pts1_cluster_label.shape[0]):
        #         for j in range(pts1_cluster_label.shape[1]):
        #             super_pixel_coordinates_1[pts1_cluster_label[i, j].cpu().numpy().item()].append((i, j))
            
        #     selected_superpixels = select_superpixels(context['image'][b, 0].permute(1, 2, 0) , super_pixel_coordinates_1  ,pts1_cluster_label)

        #     percentage = 10
            
        #     representation_gaussians_1 = []
        #     for sp in selected_superpixels:
        #         num_pixels = int(len(super_pixel_coordinates_1[sp]) * percentage / 100)
        #         representation_gaussians_1.extend(random.sample(super_pixel_coordinates_1[sp], num_pixels))
        
        #     mask = np.ones((  context['image'].shape[-2]  , context['image'].shape[-1]), dtype=bool)  
        #     for sp in selected_superpixels:
        #         for x, y in super_pixel_coordinates_1[sp]:
        #             mask[x, y] = False
        #             context['image'][b,0,:,x,y] = -1
        
        #     for x, y in representation_gaussians_1:
        #         mask[x, y] = True
        

        #     import torchvision
            
        #     torchvision.utils.save_image(context['image'][b, 0] , "del.png")
        #     plt.imshow(mask, cmap="gray")


        #     plt.savefig(f"mask_{1}.png", bbox_inches='tight')
        #     plt.close()
        
        #     mask_tensor = torch.tensor(mask, dtype=torch.bool, device=context['rep'][b][0].device)


        #     context['rep'][b][0] = mask_tensor

        
        #     super_pixel_coordinates_2 = {i: [] for i in range(pts2_cluster_label.min(), pts2_cluster_label.max() + 1)}
        #     for i in range(pts2_cluster_label.shape[0]):
        #         for j in range(pts2_cluster_label.shape[1]):
        #             super_pixel_coordinates_2[pts2_cluster_label[i, j].cpu().numpy().item()].append((i, j))
        #     representation_gaussians_2 = []
        #     mask = np.ones((   context['image'].shape[-2]  , context['image'].shape[-1]   ), dtype=bool)  
        #     for sp in selected_superpixels:
        #         if sp in super_pixel_coordinates_2:
        #             num_pixels = int(len(super_pixel_coordinates_2[sp]) * percentage / 100)
        #             representation_gaussians_2.extend(random.sample(super_pixel_coordinates_2[sp], num_pixels))
                
        #         else:
        #             print(sp , "******")
        #     for sp in selected_superpixels:
        #         if sp in super_pixel_coordinates_2:
        #             for x, y in super_pixel_coordinates_2[sp]:
        #                 mask[x, y] = False
        #                 context['image'][b,1,:,x,y] = -1

        #     for x, y in representation_gaussians_2:
        #         mask[x, y] = True

        #     # import torchvision
        #     torchvision.utils.save_image(context['image'][b, 1] , "del1.png")
        #     plt.imshow(mask, cmap="gray")


        #     plt.savefig(f"mask_{2}.png", bbox_inches='tight')
        #     plt.close()
            


        #     mask_tensor = torch.tensor(mask, dtype=torch.bool, device=context['rep'][b][1].device)

        #     # Assign to context['rep'][0][1]
        #     context['rep'][b][1] = mask_tensor



        # # # END of Masking and all

        with torch.cuda.amp.autocast(enabled=False):
           
            dec1, dec2, shape1, shape2, view1, view2   = self.backbone(context, return_views=True)
            res1 = self._downstream_head(1, [tok.float() for tok in dec1]    , shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2]   , shape2)
            if self.gs_params_head_type == 'linear':
                GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
                GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
            elif self.gs_params_head_type == 'dpt':
                GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            elif self.gs_params_head_type == 'dpt_gs':
                GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1]   , res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3] , shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2(  [tok.float() for tok in dec2]    , res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

        pts3d1 = res1['pts3d']
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2['pts3d']
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces


        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = torch.stack([GS_res1, GS_res2], dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        # Convert the features and depths into Gaussians.
        if self.pose_free:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
