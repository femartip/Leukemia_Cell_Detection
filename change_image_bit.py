# Change images from 32-bit to 8-bit mantaning the same color palette
import os
from PIL import Image
import numpy as np

path = "./data/images/256"
output_path = "./data/images/256_8bit"
for filename in os.listdir(path):
    if filename.endswith(".png"):
        image = Image.open(os.path.join(path, filename))
        image = image.convert("RGB")
        image = image.convert("P", palette=Image.ADAPTIVE, colors=256)
        image.save(os.path.join(output_path, filename))
        print(f"Image {filename} converted to 8-bit")


    