import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2

class BaMboLoader(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, image_size=None, transform=None):
        self.images_dir = images_dir
        self.images = os.listdir(images_dir)
        self.masks_dir = masks_dir
        self.masks = os.listdir(masks_dir)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size:
            image = cv2.resize(image, self.image_size)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        if mask_path.endswith((".jpg", ".png", ".tif", ".tiff")):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid mask file format")
        if self.image_size:
                mask = cv2.resize(mask, self.image_size)
        
        mask = torch.tensor(mask).long()
        #['bg','fat','cell','bone']
        mask = F.one_hot(mask, num_classes=4)
        mask = mask[:,:,2].float()
        #mask = mask.permute(2, 0, 1).float()
        
        return image, mask
    
class oursLoader(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, image_size=None, transform=None):
        self.images_dir = images_dir
        self.images = os.listdir(images_dir)
        self.masks_dir = masks_dir
        self.masks = os.listdir(masks_dir)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size:
            image = cv2.resize(image, self.image_size)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        if mask_path.endswith(".npy"):
            mask = np.load(mask_path)
        elif mask_path.endswith((".jpg", ".png")):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid mask file format")
        if self.image_size:
                mask = cv2.resize(mask, self.image_size)
        
        mask = torch.tensor(mask).float()
        if mask.max() > 1:
            mask = mask // 255
        
        return image, mask
    
class SegPC21Loader(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, image_size=None, transform=None):
        self.images_dir = images_dir
        self.images = os.listdir(images_dir)
        self.masks_dir = masks_dir
        self.masks = os.listdir(masks_dir)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size:
            image = cv2.resize(image, self.image_size)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        if mask_path.endswith(".png"):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid mask file format")
        
        if self.image_size:
                mask = cv2.resize(mask, self.image_size)
        
        mask = torch.tensor(mask).float()
        if mask.max() > 1:
            mask = mask // 255
        
        return image, mask