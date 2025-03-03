from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional
from torch_kmeans import KMeans
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
import os
import random
import time
from torch import Tensor, nn
from typing import Tuple
import torchvision
from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians , Gaussians_modified
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
def inverse_normalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean



def save_colored_pointcloud(x, y, z, r, g, b, path):
    """
    Save a point cloud to a PLY file.
    
    Args:
        x, y, z: 1D numpy arrays of shape (N,) representing point coordinates.
        r, g, b: 1D numpy arrays of shape (N,) representing color values (0-255).
        path: Output file path (string or Path object).
    """
    assert len(x) == len(y) == len(z) == len(r) == len(g) == len(b), "Input arrays must have the same length"
    
    points = np.array(list(zip(x, y, z, r, g, b)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    el = PlyElement.describe(points, 'vertex')
    PlyData([el]).write(path)
    print(f"PLY file saved to {path}")



import torch
import numpy as np
import pywt
import cv2
import random
import matplotlib.pyplot as plt
inf = float('inf')
from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor

def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
):
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


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
    prune_percent : int
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True

from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor

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
        self.prune_percent =  80
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



    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = {},
    ) -> Tuple[Gaussians, Gaussians_modified]:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        context_rep_float = context['rep'].float()
        context_rep_resized = F.interpolate(context_rep_float, size=(256, 768), mode='nearest')    
        context_rep_resized = context_rep_resized.round().int() 
        dec1_mask = context_rep_resized[:, :1] 
        dec2_mask = context_rep_resized[:, 1:]




        def select_superpixels(img, sp_cord, segments_slic, wavelet='db1', level=1, percentage=10):
                # Ensure image is on CPU and convert to numpy
            img = img.cpu().numpy()
            
            original_img = img

            if isinstance(segments_slic, torch.Tensor):
                segments_slic = segments_slic.cpu().numpy()
            
            sp_wave_values = {int(i): [] for i in sp_cord.keys()}
            
            coeffs = pywt.wavedec2(original_img[:,:,0], wavelet, level=level)
            _, (cH, cV, cD) = coeffs 
            wavelet_magnitude = np.abs(cH) + np.abs(cV) + np.abs(cD) 

            resized_z = cv2.resize(wavelet_magnitude, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            for i in range(segments_slic.shape[0]):
                for j in range(segments_slic.shape[1]):
                    key = int(segments_slic[i, j])  # Ensure integer key
                    if key in sp_wave_values:
                        sp_wave_values[key].append(abs(resized_z[i, j]))
                    else:
                        print(f"Warning: Key {key} not found in sp_wave_values")
            mean_wavelet_values = np.array([np.nanmean(sp_wave_values[k]) if np.any(~np.isnan(sp_wave_values[k])) else 0 for k in sp_wave_values.keys()])

            
            left = int(len(sp_cord.keys()) * self.prune_percent / 100)
            right = len(sp_cord.keys()) - left 
            indices = np.argsort(mean_wavelet_values)

            # selected_superpixels = np.array(list(sp_cord.keys()))[indices[:left]]
            # Random picking
            selected_superpixels = np.random.choice(np.array(list(sp_cord.keys())), left, replace=False)
            # selected_superpixels_high =   np.array(list(sp_cord.keys()))[indices[-right:]]  
            selected_superpixels_high = []

            return selected_superpixels , selected_superpixels_high




        with torch.cuda.amp.autocast(enabled=False):

            dec1, dec2, shape1, shape2, view1, view2   = self.backbone(context, return_views=True)
            res1 = self._downstream_head(1, [tok.float() for tok in dec1]  +  [tok.float() for tok in dec1_mask]  , shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2] +  [tok.float() for tok in dec2_mask]  , shape2)
            if self.gs_params_head_type == 'linear':
                GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
                GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
            elif self.gs_params_head_type == 'dpt':
                GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1]  +  [tok.float() for tok in dec1_mask]   , shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2] +   [tok.float() for tok in dec2_mask], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            elif self.gs_params_head_type == 'dpt_gs':
                GS_res1 = self.gaussian_param_head(  [tok.float() for tok in dec1] +  [tok.float() for tok in dec1_mask]   , res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3] , shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2(  [tok.float() for tok in dec2]   + [tok.float() for tok in dec2_mask]   , res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
        
        pts3d1 = res1['pts3d']
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2['pts3d']
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces


# 300 clusters masking

        for b in range(context['image'].size(0)):  # Iterate over batch

            pts1_append = torch.cat((res1['pts3d'][b], context['image'][b, 0].permute(1, 2, 0)), dim=-1)
            pts2_append = torch.cat((res2['pts3d'][b], context['image'][b, 1].permute(1, 2, 0)), dim=-1)

            # Flatten into a (N, 6) tensor
            tensor_data = torch.cat((pts1_append, pts2_append), dim=0).reshape(-1, 6)

            # Ensure no gradients are tracked
            tensor_data = tensor_data.detach()

            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor_data = tensor_data.to(device)

            # Reshape to match KMeans expected input shape (BS, N, D)
            tensor_data = tensor_data.unsqueeze(0)  # Adds batch dimension (1, N, D)

            kmeans = KMeans(n_clusters=300, mode='euclidean', verbose=0)
    
            # Perform clustering
            start  = time.time()
            cluster_result = kmeans.fit_predict(tensor_data)  # Returns a ClusterResult object

            print(f"##CLUSTER {time.time()-start}")
            cluster_labels = cluster_result   

            # Reshape cluster labels back to original dimensions
            num_pts1 = pts1_append.shape[0] * pts1_append.shape[1]  # Total points in first image
            pts1_cluster_label = cluster_labels[0][:num_pts1].reshape(pts1_append.shape[0], pts1_append.shape[1])
            pts2_cluster_label = cluster_labels[0][num_pts1:].reshape(pts2_append.shape[0], pts2_append.shape[1])

            # Ensure labels are on the correct device
            pts1_cluster_label = pts1_cluster_label.to(device)
            pts2_cluster_label = pts2_cluster_label.to(device)


            super_pixel_coordinates_1 = {i: [] for i in range(pts1_cluster_label.min(), pts1_cluster_label.max() + 1)}
            
            for i in range(pts1_cluster_label.shape[0]):
                for j in range(pts1_cluster_label.shape[1]):
                    super_pixel_coordinates_1[pts1_cluster_label[i, j].cpu().numpy().item()].append((i, j))
            
            selected_superpixels , selected_superpixels_high = select_superpixels(context['image'][b, 0].permute(1, 2, 0) , super_pixel_coordinates_1  ,pts1_cluster_label , percentage=self.prune_percent)
  
            
            representation_gaussians_1 = []
            for sp in selected_superpixels:
                num_pixels = int(len(super_pixel_coordinates_1[sp]) * 10 / 100)
                representation_gaussians_1.extend(random.sample(super_pixel_coordinates_1[sp], num_pixels))

            mask = np.ones((  context['image'].shape[-2]  , context['image'].shape[-1]), dtype=bool)  
            for sp in selected_superpixels:
                for x, y in super_pixel_coordinates_1[sp]:
                    mask[x, y] = False
                    

            for x, y in representation_gaussians_1:
                mask[x, y] = True


            
            # torchvision.utils.save_image(context['image'][b, 0] , "del.png")
            # plt.imshow(mask, cmap="gray")


            # plt.savefig(f"mask_3d_{1}.png", bbox_inches='tight')
            # plt.close()

            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=context['rep'][b][0].device)


            context['rep'][b][0] = mask_tensor


            super_pixel_coordinates_2 = {i: [] for i in range(pts2_cluster_label.min(), pts2_cluster_label.max() + 1)}
            for i in range(pts2_cluster_label.shape[0]):
                for j in range(pts2_cluster_label.shape[1]):
                    super_pixel_coordinates_2[pts2_cluster_label[i, j].cpu().numpy().item()].append((i, j))
            representation_gaussians_2 = []
            mask = np.ones((   context['image'].shape[-2]  , context['image'].shape[-1]   ), dtype=bool) 
            selected_superpixels , selected_superpixels_high = select_superpixels(context['image'][b, 1].permute(1, 2, 0) , super_pixel_coordinates_2  ,pts2_cluster_label , percentage=self.prune_percent) 
            for sp in selected_superpixels:
                if sp in super_pixel_coordinates_2:
                    num_pixels = int(len(super_pixel_coordinates_2[sp]) * 10 / 100)
                    representation_gaussians_2.extend(random.sample(super_pixel_coordinates_2[sp], num_pixels))
                
                else:
                    print(sp , "******")

            for sp in selected_superpixels:
                if sp in super_pixel_coordinates_2:
                    for x, y in super_pixel_coordinates_2[sp]:
                        mask[x, y] = False
                        

            for x, y in representation_gaussians_2:
                mask[x, y] = True





            # torchvision.utils.save_image(context['image'][b, 1] , "del1.png")
            # plt.imshow(mask, cmap="gray")


            # plt.savefig(f"mask_3d_{2}.png", bbox_inches='tight')
            # plt.close()
            


            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=context['rep'][b][1].device)

            # Assign to context['rep'][0][1]
            context['rep'][b][1] = mask_tensor
            



        pts3d1 = res1['pts3d']
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2['pts3d']
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces


#####

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



        # rep = context['rep']
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


        
        # gaussians.scales  = gaussians.scales  * rep.view(b, 2, 65536, 1, 1, 1)  
        # export_ply(gaussians.means.reshape(-1,3), gaussians.scales.reshape(-1,3), gaussians.rotations.reshape(-1,4), gaussians.harmonics.reshape(-1,3,25), gaussians.opacities.reshape(-1), path=Path('/workspace/raid/cdsbad/splat3r_try/NoPoSplat/mask.ply'))

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

    # means: Float[Tensor, "*batch 3"]
    # covariances: Float[Tensor, "*batch 3 3"]
    # scales: Float[Tensor, "*batch 3"]
    # rotations: Float[Tensor, "*batch 4"]
    # harmonics: Float[Tensor, "*batch 3 _"]
    # opacities: Float[Tensor, " *batch"]



        return   Gaussians(
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

        ),     Gaussians_modified(
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
            rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            
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