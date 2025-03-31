import torch
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
from cellSAM.sam_inference import CellSAM
import torch.nn.functional as F
from inference import segment_image

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def postprocess(mask, circularity_threshold=0.7):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area > 1000: 
        return False

    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    try:
        ellipse = cv2.fitEllipse(contour)
        ellipse_mask = np.zeros_like(mask)
        ellipse_mask = cv2.ellipse(ellipse_mask, ellipse, 1, -1)
        intersection = np.logical_and(mask, ellipse_mask).sum()
        union = np.logical_or(mask, ellipse_mask).sum()
        iou = intersection / union if union > 0 else 0
    except:
        return circularity > circularity_threshold
        
    return circularity > circularity_threshold  or iou > 0.85

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


def draw_masks(image, masks_generated, labels):
    masked_image = image.copy()
    
    for i in range(masks_generated.shape[0]):
        mask = masks_generated[i]
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
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_polygons = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [[int(X + point[0][0]), int(Y + point[0][1])] for point in approx]    
        polygon.append(polygon[0])      # Append the first point to close the polygon
        mask_polygons.append(polygon)
    return mask_polygons

def main(dataset_name, num_images=None):
    if dataset_name == "BaMbo":
        dir_path = "./data/BaMbo/test/images"
        results_file = "BaMbo"
    elif dataset_name == "ours":
        #dir_path = "./data/ours/512/images/"
        dir_path = "./data/ours/22B0019275_a1_CD34/images"  
        results_file = "22B0019275_a1_CD34"      

    config = {
        'enc_layers': 6,
        'dec_layers': 6,
        'dim_feedforward': 1024,
        'hidden_dim': 256,
        'dropout': 0.0,
        'nheads': 8,
        'num_query_position': 3500,
        'num_query_pattern': 1,
        'spatial_prior': 'learned',
        'attention_type': 'RCDA',
        'num_feature_levels': 1,
        'device': 'cuda',
        'seed': 42,
        'num_classes': 2
    }
    sam_model = CellSAM(config)
    sam_model.load_state_dict(torch.load('./models/cellsam-base.pt'))
    #sam_model.load_state_dict(torch.load('./models/finetuned_bone_marrow_cellsam.pth'))

    sam_model = sam_model.to(DEVICE)

    images = []
    images_names = []
    image_coordinates = []

    logging.info("Reading images")
    for file in os.listdir(dir_path):
        if file.endswith((".png", ".jpg")):
            images_names.append(str(file))
            image_bgr = cv2.imread(os.path.join(dir_path, file))
            x,y = read_image_metadata(os.path.join(dir_path, file))
            images.append(image_bgr)
            image_coordinates.append((x, y))
    
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
        try:
            masks, _, bboxes = segment_image(image, sam_model, device=DEVICE)
        except Exception as e:
            logging.warning(f"Error processing image {images_names[i]}: {e}")
            continue

        num_cls = len(np.unique(masks).tolist()) - 1
        if num_cls == 0:
            logging.warning(f"No cells detected in image {images_names[i]}")
            continue
        masks = np.clip(masks, 0, num_cls - 1)  # Ensure mask values are within valid range
        masks_one_hot = F.one_hot(torch.tensor(masks).long(), num_classes=num_cls)
        masks_one_hot = masks_one_hot.permute(2,0,1).numpy()
        masks_one_hot = masks_one_hot[1:] # Remove background

        masks_one_hot = np.array([mask for mask in masks_one_hot if postprocess(mask)])

        labels_roi = []
        for j,mask in enumerate(masks_one_hot):
            roi = region_of_interest(image, mask)
            #cv2.imwrite(f"./CellSAM/bin_masks/mask_{i}_{j}.png", roi)
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
            
        masked_image = draw_masks(image, masks_one_hot, labels_roi)
        cv2.imwrite(f"./results/{results_file}/{images_names[i]}", masked_image)
        logging.info(f"Image {images_names[i]} processed")

    logging.info("Saving geojson")
    out_file = open(f'results/{results_file}.geojson', 'w')
    json.dump(geojson, out_file, indent=4)
    out_file.close()

    logging.info("Merging overlapping polygons")
    with open(f'results/{results_file}.geojson') as f:
        geojson = json.load(f)
    merged_features = merge_overlapping_polygons.merge_overlapping_polygons(geojson)
    geojson['features'] = merged_features
    out_file = open(f'results/{results_file}.geojson', 'w')
    json.dump(geojson, out_file, indent=4)
    out_file.close()

    logging.info("Done")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract images from a directory, generate masks and classify them. Results are saved in a geojson file so that they can be loaded to qupath")
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('--num_images', type=int, help='Number of images to process')
    args = parser.parse_args()

    init_time = time.time()

    main(args.dataset, args.num_images)

    end_time = time.time()
    logging.info(f"Total time: {(end_time - init_time)/3600} hours")