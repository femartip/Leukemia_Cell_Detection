import cv2
import numpy as np
import os 
import torch.nn.functional as F
import torch

def draw_masks(image, masks_generated, labels):
    masked_image = image.copy()
    
    if len(masks_generated.shape) == 2:
        masks_generated = masks_generated.unsqueeze(0)

    for i in range(masks_generated.shape[0]):
        mask = masks_generated[i]
        print(mask.shape)
        if labels[i] == 0:
            color = [0, 255, 0]
        elif labels[i] == 1:
            color = [0, 0, 255]
        else:
            color = [255, 255, 255]

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
            mask = np.repeat(mask, 3, axis=2)
        elif len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
            mask = np.repeat(mask, 3, axis=2)
        
        masked_image = np.where(mask, np.array(color), masked_image)
    overlay = cv2.addWeighted(image.astype(np.uint8), 0.7, masked_image.astype(np.uint8), 0.3, 0)
    return overlay


if __name__ == '__main__':
    imges_path = './data/ours/512_small/images/'
    masks_path = './data/ours/512_small/masks/img/'
    overlay_path = './data/ours/512_small/overlay/'

    for img_path in os.listdir(imges_path):
        if img_path.endswith((".jpg", ".png", ".tif", ".tiff")):
            image = cv2.imread(os.path.join(imges_path, img_path))
            mask = cv2.imread(os.path.join(masks_path, img_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            cls = np.unique(mask).shape[0]
            if cls > 2:
                mask_one_hot = F.one_hot(torch.tensor(mask).long(), num_classes=cls)
                mask = mask_one_hot
            else:
                mask = torch.tensor(mask).long()

            overlay = draw_masks(image, mask, [0, 1])
            cv2.imwrite(os.path.join(overlay_path, img_path), overlay)
