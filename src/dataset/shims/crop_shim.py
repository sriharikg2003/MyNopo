import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews
import matplotlib.pyplot as plt
import numpy as np
import random 
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float



# mask_generator = SAM2AutomaticMaskGenerator(sam2)
# SAM END


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)
import random

def choose_and_get_patch_coordinates(images):
    h, w = 256, 256
    ph, pw = 64, 64
    max_x = w - pw
    max_y = h - ph


    # x1 = random.randint(0, max_x)
    # y1 = random.randint(0, max_y)

    x1 , y1 = 100 , 100

    while True:
        x2 = random.randint(0, max_x)
        y2 = random.randint(0, max_y)
        if not (x1 < x2 + pw and x2 < x1 + pw and y1 < y2 + ph and y2 < y1 + ph):
            break

    row_start1, row_end1 = y1, y1 + ph
    col_start1, col_end1 = x1, x1 + pw

    row_start2, row_end2 = y2, y2 + ph
    col_start2, col_end2 = x2, x2 + pw


    images[:, :, row_start1 : row_end1 ,   col_start1 : col_end1 ] = 0
    # images[:, :, row_start2 : row_end2 ,   col_start2 : col_end2 ] = 0
    return (row_start1, row_end1, col_start1, col_end1), (row_start2, row_end2, col_start2, col_end2)

# def get_superpixel_representation(images):

#     img = images.permute(0, 2, 3, 1).cpu().numpy()  
#     batch_masks = []
    
#     for b in range(img.shape[0]):  
#         segments_slic = slic(img[b], n_segments=300, compactness=10, sigma=1, start_label=1)
        
#         super_pixel_coordinates = {i: [] for i in range(segments_slic.min(), segments_slic.max() + 1)}
        
#         for i in range(segments_slic.shape[0]):
#             for j in range(segments_slic.shape[1]):
#                 super_pixel_coordinates[segments_slic[i, j]].append((i, j))
        
#         selected_superpixels = random.sample(list(super_pixel_coordinates.keys()), 4)

#         percentage = 10
#         representation_gaussians = []
#         for sp in selected_superpixels:
#             num_pixels = int(len(super_pixel_coordinates[sp]) * percentage / 100)
#             representation_gaussians.extend(random.sample(super_pixel_coordinates[sp], num_pixels))
        
        
#         mask = np.ones((img.shape[1], img.shape[2]), dtype=bool)  


#         for sp in selected_superpixels:
#             for x, y in super_pixel_coordinates[sp]:
#                 mask[x, y] = False
#                 img[b][x,y,:] = 0

#         for x, y in representation_gaussians:
#             mask[x, y] = True

#         # plt.figure(figsize=(6, 3))
#         # plt.subplot(1, 2, 1)
#         # plt.imshow(img[b])
#         # plt.title("Modified Image")
#         # plt.axis("off")

#         # # Save the mask
#         # plt.subplot(1, 2, 2)
#         # plt.imshow(mask, cmap="gray")
#         # plt.title("Mask")
#         # plt.axis("off")

#         # plt.savefig(f"mask_{b}.png", bbox_inches='tight')
#         # plt.close()

#         batch_masks.append(torch.tensor(mask))  # Convert to PyTorch tensor
#     images = torch.tensor(img).permute(0, 3, 1, 2) 
#     return images , torch.stack(batch_masks)  


# Same area patches 

import torch
import logging
import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

def get_sam(images):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2_checkpoint = "//workspace/raid/cdsbad/splat3r_try/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "//workspace/raid/cdsbad/splat3r_try/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    img = images.permute(0, 2, 3, 1).cpu().numpy()  
    batch_masks = []
    
    for i, single_img in enumerate(img):
        print("************************GOOL")
        masks = mask_generator.generate(single_img)
        print("************************GOOL")
        mask = masks[0]['segmentation']  
        print("************************GOOL")
        batch_masks.append(torch.tensor(mask, dtype=torch.float32))
        
        # Save the mask image
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(f"mask_{i}.png")

    images = torch.tensor(img).permute(0, 3, 1, 2) 
    return images, torch.stack(batch_masks)




def get_superpixel_representation(images):

    img = images.permute(0, 2, 3, 1).cpu().numpy()  
    batch_masks = []
    
    segments_slic = slic(img[0], n_segments=300, compactness=10, sigma=1, start_label=1)
    
    super_pixel_coordinates = {i: [] for i in range(segments_slic.min(), segments_slic.max() + 1)}
    
    for i in range(segments_slic.shape[0]):
        for j in range(segments_slic.shape[1]):
            super_pixel_coordinates[segments_slic[i, j]].append((i, j))
    
    selected_superpixels = random.sample(list(super_pixel_coordinates.keys()), 150)
    percentage = 10
    representation_gaussians = []
    for sp in selected_superpixels:
        num_pixels = int(len(super_pixel_coordinates[sp]) * percentage / 100)
        representation_gaussians.extend(random.sample(super_pixel_coordinates[sp], num_pixels))
    for b in range(img.shape[0]):  
        
        
        mask = np.ones((img.shape[1], img.shape[2]), dtype=bool)  


        # for sp in selected_superpixels:
        #     for x, y in super_pixel_coordinates[sp]:
        #         mask[x, y] = False
        #         img[b][x,y,:] = 0

        # for x, y in representation_gaussians:
        #     mask[x, y] = True

        # plt.figure(figsize=(6, 3))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img[b])
        # plt.title("Modified Image")
        # plt.axis("off")

        # # Save the mask
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask, cmap="gray")
        # plt.title("Mask")
        # plt.axis("off")

        # plt.savefig(f"mask_{b}.png", bbox_inches='tight')
        # plt.close()

        batch_masks.append(torch.tensor(mask))  # Convert to PyTorch tensor
    images = torch.tensor(img).permute(0, 3, 1, 2) 
    return images , torch.stack(batch_masks)  



def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int] , is_context : bool) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    old_images = images.clone()
    if is_context:
        # Patch

        # (row_start1, row_end1, col_start1, col_end1), (row_start2, row_end2, col_start2, col_end2) = choose_and_get_patch_coordinates(images)

        # return {
        #     **views,
        #     "image": images,
        #     "intrinsics": intrinsics,
        #     "patch" : (row_start1, row_end1, col_start1, col_end1 , row_start2, row_end2, col_start2, col_end2)

        # }


        # Super pixels
        new_images , representation_gaussians = get_superpixel_representation(images)

        return {
            **views,
            "image": new_images,
            "intrinsics": intrinsics,
            "rep" : representation_gaussians,
            "original" : old_images

        }

    else:

        return {
            **views,
            "image": images,
            "intrinsics": intrinsics,
        }




def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape , is_context=True),
        "target": apply_crop_shim_to_views(example["target"], shape , is_context=False),
    }
