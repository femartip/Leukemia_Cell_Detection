import os
import cv2

mask_path = "data/masks_sam_h/"
image_path = "data/images/256/"
save_images = "data/images/images"

for file in os.listdir(mask_path):
    mask_name = file.split(".")[0]
    corresponding_image = cv2.imread(os.path.join(image_path, mask_name + ".png"))
    cv2.imwrite(os.path.join(save_images, mask_name + ".png"), corresponding_image)