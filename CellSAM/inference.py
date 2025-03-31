import torch
from torch import nn
import numpy as np
from scipy.ndimage import binary_fill_holes, find_objects

def fill_holes_and_remove_small_masks(masks, min_size=15):
    #https://github.com/MouseLand/cellpose/blob/509ffca33737058b0b4e2e96d506514e10620eb3/cellpose/utils.py#L616
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim
        )

    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:
                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = j + 1
                j += 1
    return masks

def format_image_shape(img):
    if len(img.shape) == 2:
        img = img[..., None]
    elif len(img.shape) == 3:
        channel_idx = np.argmin(img.shape)
        img = np.transpose(
            img, [i for i in range(len(img.shape)) if i != channel_idx] + [channel_idx]
        )
    else:
        raise ValueError(
            "Invalid image shape. Only (H, W) or (H, W, C) images are supported."
        )
    H, W, C = img.shape
    if C > 3:
        raise ValueError("A maximum of three channels is supported.")

    out_img = np.zeros((H, W, 3))
    out_img[:, :, -C:] = img
    return out_img

def segment_image(
    img: np.ndarray,
    model: nn.Module = None,
    bounding_boxes: list[list[float]] = None,
    bbox_threshold: float = 0.2,
    device: str = "cpu",
):
    if bounding_boxes is not None:
        bounding_boxes = torch.tensor(bounding_boxes).unsqueeze(0)
        assert (
            len(bounding_boxes.shape) == 3
        ), "Bounding boxes should be of shape (number of boxes, 4)"

    model.bbox_threshold = bbox_threshold

    img = format_image_shape(img)
    img = img.transpose((2, 0, 1))  # channel first for pytorch.
    img = torch.from_numpy(img).float().unsqueeze(0)

    if "cuda" in str(device):
        model, img = model.to(device), img.to(device)

    preds = model.predict(img, x=None, boxes_per_heatmap=bounding_boxes, device=device)
    if preds is None:
        print("No cells detected in the image.")
        return np.zeros(img.shape[1:], dtype=np.int32), None, None

    segmentation_predictions, _, x, bounding_boxes = preds

    mask = fill_holes_and_remove_small_masks(segmentation_predictions, min_size=25)

    return mask, x.cpu().numpy(), bounding_boxes