import cv2 
import os
import numpy as np


file_path = "./data/SegPC-21/validation/masks/masks"
images_path = "./data/SegPC-21/validation/images"
output_path = "./data/SegPC-21/validation/masks/img"

img_list = sorted(os.listdir(images_path))
files_list = sorted(os.listdir(file_path))

for img in img_list:
    if img.endswith(".bmp"):
        img_name = img.split(".")[0]
        img_file = cv2.imread(os.path.join(images_path, img), cv2.IMREAD_COLOR)
        img_file = cv2.resize(img_file, (1024, 1024))
        
        masks = [mask for mask in files_list if mask.startswith(img_name)]
        bin_mask = np.zeros((img_file.shape[0], img_file.shape[1]), dtype=np.uint8)
        for mask in masks:
            mask_img = cv2.imread(os.path.join(file_path, mask), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (1024, 1024))
            mask_img[mask_img != 40] = 0
            mask_img[mask_img == 40] = 255
            try:
                bin_mask = cv2.bitwise_or(bin_mask, mask_img)
            except:
                print(mask)
                print(mask_img.shape)
                print(bin_mask.shape)
                continue

        
        cv2.imwrite(os.path.join(output_path, img.replace(".bmp", ".png")), bin_mask)
        #cv2.imwrite(os.path.join("./data/SegPC-21/validation/images", img.replace(".bmp", ".png")), img_file)

print("Done")

            
