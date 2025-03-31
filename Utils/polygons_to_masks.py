import json
import numpy as np
import cv2
from shapely.geometry import Polygon
import os

def create_binary_mask_from_json(json_data, output_path=None, visualize=True):
    height = json_data['height']
    width = json_data['width']

    mask = np.zeros((height, width), dtype=np.uint8)
    for box in json_data['boxes']:
        if box['type'] == 'polygon':
            points = np.array(box['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    
    return mask


if __name__ == "__main__":
    dir_path = "./data/ours/512_small/masks/data"
    output_path = "./data/ours/512_small/masks/img"
    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            output = os.path.join(output_path, file.replace(".json", ".png"))
            with open(os.path.join(dir_path, file), "r") as f:
                json_data = json.load(f)
            mask = create_binary_mask_from_json(json_data)
            cv2.imwrite(output, mask)
            print(f"Mask saved to {output}")
