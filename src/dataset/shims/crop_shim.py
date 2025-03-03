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


import torch
import numpy as np
import pywt
import random
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import cv2
def get_wavelet_superpixel_representation(images, wavelet='db1', level=1, percentage=10):
    img = images.permute(0, 2, 3, 1).cpu().numpy()
    batch_masks = []

    for b in range(img.shape[0]): 

        mask = np.ones((img.shape[1], img.shape[2]), dtype=bool)

        batch_masks.append(torch.tensor(mask))  

    images = torch.tensor(img).permute(0, 3, 1, 2)  
    return images, torch.stack(batch_masks)




# Same area patches 

import torch
import logging
import multiprocessing

def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int] , is_context : bool) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    old_images = images.clone()
    if is_context:



        # Super pixels
        new_images , representation_gaussians = get_wavelet_superpixel_representation(images)

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
