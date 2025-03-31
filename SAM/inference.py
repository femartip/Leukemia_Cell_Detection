import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from dataset import ImageDataLoader, SAMDataset
import torch
from transformers import SamModel, SamProcessor
import os

def inference(data_dir):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # let's take a random training example
  idx = 10
  dataset = ImageDataLoader(images_dir=os.path.join(data_dir,"images"), masks_dir=os.path.join(data_dir,"masks"), image_size=(512, 512))
  # load image
  image, mask = dataset[idx]
  # get box prompt based on ground truth segmentation map
  #prompt = get_bounding_box(mask)
  prompt = [0,0,512,512]  
  # prepare image + box prompt for the model

  processor = SamProcessor.from_pretrained("./models/cellsam_base_v1.1.pt")
  model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
  inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
  for k,v in inputs.items():
    print(k,v.shape)

  model.eval()

  # forward pass
  with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)
  print(outputs.pred_masks.shape)
  # apply sigmoid
  medsam_seg_prob = torch.sigmoid(outputs.pred_masks)
  print(medsam_seg_prob.shape)
  # convert soft mask to hard mask
  medsam_seg_prob = medsam_seg_prob.cpu().numpy()
  medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
  print(medsam_seg.shape)

  plot(image, medsam_seg, mask)

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    #h, w = mask.shape[-2:]
    #mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask.squeeze(0)  #shape (1, 256, 256)
    print(mask_image.shape)
    print(type(mask_image))
    plt.imsave("./mask.png", mask_image)


def plot(image, medsam_seg, ground_truth_mask):
  #fig, axes = plt.subplots()

  image = image.permute(1, 2, 0).numpy()

  mask_image = medsam_seg.squeeze(0)  #shape (1, 256, 256)
  print(mask_image.shape)
  print(type(mask_image))
  plt.imsave("./mask.png", mask_image)

  gt_mask = ground_truth_mask.argmax(0).squeeze(0)
  print(gt_mask.shape)
  print(type(gt_mask))
  plt.imsave("./gt_mask.png", gt_mask)
  #medsam_seg = np.squeeze(medsam_seg, 2) 
  #ground_truth_mask = np.squeeze(ground_truth_mask, 2)

  #plt.imsave("./image.png", np.array(image))
  #show_mask(medsam_seg)

  #axes.title.set_text(f"Predicted mask")
  #axes.axis("off")

  #Compare this to the ground truth segmentation:
  #fig, axes = plt.subplots()

  #axes.imshow(np.array(image))
  #show_mask(ground_truth_mask, axes)
  #axes.title.set_text(f"Ground truth mask")
  #axes.axis("off")

if __name__ == "__main__":
   inference(data_dir="./data/BaMbo/train/")