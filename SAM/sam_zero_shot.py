import torch
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import os
import cv2
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import json
from copy import deepcopy
import logging
import argparse
import time
import merge_overlapping_polygons

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model():
    num_classes = 1
    #Load the model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    #Modify the models head
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model
    
def area_of_mask(mask):
    return np.sum(mask)

def region_of_interest(image, binary_mask):
    result = image.copy()
    result[binary_mask == 0] = 0
    result[binary_mask != 0] = image[binary_mask != 0]
    return result

def filter_masks(masks, min_area=100, max_area=2000):
    filtered_masks = []
    for i in range(len(masks)): 
        binary_mask = masks[i]['segmentation'].astype(np.uint8)
        area = area_of_mask(binary_mask)
        if area > min_area and area < max_area:
            filtered_masks.append(binary_mask)
    return filtered_masks

def draw_masks_fromDict(image, masks_generated, labels):
    masked_image = image.copy()
    for i in range(len(masks_generated)):
        mask = masks_generated[i]
        if labels[i] == 0:
            color = [0, 255, 0]
        elif labels[i] == 1:
            color = [0, 0, 255]
        else:
            color = [255, 255, 255]
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        masked_image = np.where(mask, np.array(color), masked_image)
    overlay = cv2.addWeighted(image.astype(np.uint8), 0.7, masked_image.astype(np.uint8), 0.3, 0)
    return overlay

def preprocessing(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]) 
    img = cv2.filter2D(image, -1, kernel)
    img = cv2.convertScaleAbs(img, alpha=1.35, beta=10)
    return img

def read_image_metadata(image_path):
    img = Image.open(image_path)
    metadata = img.info
    x = metadata.get("x")
    y = metadata.get("y")
    return x, y

def mask_to_polygons(mask, coords):
    X,Y = coords
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_polygons = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [[int(X + point[0][0]), int(Y + point[0][1])] for point in approx]    
        polygon.append(polygon[0])      # Append the first point to close the polygon
        mask_polygons.append(polygon)
    return mask_polygons

def main(dir_path, model_type="vit_h", num_images=None):
    if model_type == "vit_b":
        sam = sam_model_registry[model_type](checkpoint="./models/sam_vit_b_01ec64.pth")
    elif model_type == "vit_h":
        sam = sam_model_registry[model_type](checkpoint="./models/sam_vit_h_4b8939.pth")
    else:
        raise ValueError(f"Model type {model_type} not supported")
    sam.to(DEVICE)

    images = []
    images_names = []
    image_coordinates = []

    logging.info("Reading images")
    for file in os.listdir(dir_path):
        if file.endswith(".png"):
            images_names.append(str(file))
            image_bgr = cv2.imread(os.path.join(dir_path, file))
            x,y = read_image_metadata(os.path.join(dir_path, file))
            images.append(image_bgr)
            image_coordinates.append((x, y))

    mask_generator = SamAutomaticMaskGenerator(sam)
    #mask_predictor = SamPredictor(sam)
    
    classifier_model = get_model()
    classifier_model.load_state_dict(torch.load("./models/bin_mask_classifier.pth"))
    classifier_model.eval()
    classifier_model.to(DEVICE)

    geojson_template = {
                "type": "Feature",
                "geometry": {
                    "type":"Polygon",
                    "coordinates":[]
                },
                "properties":{
                    "objectType":"annotation",
                    "classification":{"name":"","color":[]}
                }
    }

    geojson = {"type": "FeatureCollection",
        "features": []
    }

    if not num_images:
        max_img = len(images)
    else:
        max_img = num_images

    logging.info("Generating masks")

    for i in range(max_img):
        image = images[i]
        masks = mask_generator.generate(image)

        filtered_masks = filter_masks(masks)
        labels_roi = []
        for j,mask in enumerate(filtered_masks):
            roi = region_of_interest(image, mask)
            cv2.imwrite(f"./data/bin_masks/mask_{i}_{j}.png", roi)
            roi = torch.tensor(roi).permute(2,0,1).unsqueeze(0).float()
            roi = roi.to(DEVICE)
            output = classifier_model(roi)
            output = output.cpu().detach().numpy()
            output = 1 if output[0][0] > 0.5 else 0
            labels_roi.append(output)

            X,Y = int(image_coordinates[i][0]), int(image_coordinates[i][1])
            polygons_mask = mask_to_polygons(mask, (X,Y))
            for k, polygon in enumerate(polygons_mask):
                if len(polygon) < 3:
                    logging.warning(f"Polygon {k} has less than 3 coordinates")
                else:
                    geojson_polygon = deepcopy(geojson_template)
                    geojson_polygon["geometry"]["coordinates"] = [polygon]
                    geojson_polygon["properties"]["classification"]["name"] = "Positive" if output == 1 else "Negative"
                    geojson_polygon["properties"]["classification"]["color"] = [0, 255, 0] if output == 1 else [0, 0, 255]
                    geojson["features"].append(geojson_polygon)
        
        masked_image = draw_masks_fromDict(image, filtered_masks, labels_roi)
        cv2.imwrite(f"./data/masks/{images_names[i]}", masked_image)
        logging.info(f"Image {images_names[i]} processed")

    logging.info("Saving geojson")
    out_file = open('data/annotations.geojson', 'w')
    json.dump(geojson, out_file, indent=4)
    out_file.close()

    logging.info("Merging overlapping polygons")
    with open('data/annotations.geojson') as f:
        geojson = json.load(f)
    merged_features = merge_overlapping_polygons.merge_overlapping_polygons(geojson)
    geojson['features'] = merged_features
    out_file = open('data/annotations.geojson', 'w')
    json.dump(geojson, out_file, indent=4)
    out_file.close()

    logging.info("Done")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract images from a directory, generate masks and classify them. Results are saved in a geojson file so that they can be loaded to qupath")
    parser.add_argument('dir_path', type=str, help='Path to the input images directory')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of model to use (default: vit_h)')
    parser.add_argument('--num_images', type=int, help='Number of images to process (default: 500)')
    args = parser.parse_args()

    init_time = time.time()

    main(args.dir_path, args.model_type, args.num_images)

    end_time = time.time()
    logging.info(f"Total time: {(end_time - init_time)/3600} hours")