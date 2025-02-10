
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
import open3d as o3d
import numpy as np
import torch

def export_gaussian_splats_to_ply(gaussians_means, gaussians_covariances, gaussians_opacities, output_filename):
    """
    Exports the Gaussian means as splats (points with radii and opacities) to a PLY file.
    """
    # Flatten the means to create an Nx3 array of points (N = number of Gaussians)
    points = gaussians_means.view(-1, 3).cpu().numpy()
    
    # Flatten the covariance matrices (to get radii)
    covariances = gaussians_covariances.view(-1, 3, 3).cpu().numpy()

    # Flatten opacities (to scale the color or opacity of the splats)
    opacities = gaussians_opacities.view(-1).cpu().numpy()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Compute radii based on the eigenvalues of the covariance matrix
    radii = []
    for cov in covariances:
        eigvals, _ = np.linalg.eigh(cov)  # Eigenvalues for each covariance matrix
        radius = np.max(np.sqrt(eigvals))  # We take the largest eigenvalue for scaling
        radii.append(radius)
    radii = np.array(radii)

    # Create a list of colors based on opacities (we can use a colormap or just use grayscale)
    colors = np.ones((points.shape[0], 3))  # Default white color
    colors *= np.expand_dims(opacities, axis=1)  # Modulate color intensity by opacity

    # Set colors in point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # To visualize the Gaussian splats, we can simulate a "splat" using small spheres
    # Create spheres at each point with radius corresponding to the Gaussian size
    spheres = []
    for i, point in enumerate(points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radii[i] * 0.1)  # Scale factor to control size
        sphere.translate(point)
        spheres.append(sphere)

    # Combine all the spheres into a single mesh (optional)
    splat_mesh = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        splat_mesh += sphere

    # Optionally, visualize or save the splats
    o3d.io.write_triangle_mesh(output_filename, splat_mesh)
    print(f"Gaussian splats saved to {output_filename}")




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


        gaussians.means = gaussians_means_reshaped.view(b,2*h*w , 3)
        gaussians.covariances = gaussians_covariances_reshaped.view(b,2*h*w , 3,3)
        gaussians.harmonics = gaussians_harmonics_reshaped.view(b,2*h*w , 3,-1)
        gaussians.opacities = gaussians_opacities_reshaped.view(b,2*h*w)

        export_gaussian_splats_to_ply(gaussians.means, gaussians.covariances, gaussians.opacities, "output_gaussian_splats.ply")


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



        exit()
        # import torchvision
        # torchvision.utils.save_image(depth[0][1] , f"depth_{stride}.png")
        # torchvision.utils.save_image(color[0][1] , f"color_{stride}.png")
        return DecoderOutput(color, depth)