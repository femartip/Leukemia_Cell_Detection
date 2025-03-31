import openslide
import os
import numpy as np
from PIL import Image, PngImagePlugin, ImageOps, ImageDraw, ImageFont
import argparse
from paquo.projects import QuPathProject
from paquo.projects import QuPathProjectImageEntry
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString
import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def extract_annotations(qp_img: QuPathProjectImageEntry, x_img:int, y_img:int, width_img:int, height_img:int)->Image:
    image = qp_img
    annotations = image.hierarchy.annotations

    possible_label = {"Mieloblasto CD34 negativo": 1, "Mieloblasto CD34 Positivo": 2, "Capilar cd34" :3, "Tejido graso": 4, "Eritroblasto": 5, "Trabecula Osea": 6, "Granulocito": 7, "Region*": 0, "Otros": 0}
    label_colour = {"Mieloblasto CD34 negativo": (255, 0, 0), "Mieloblasto CD34 Positivo": (0, 255, 0), "Capilar cd34" :(0, 0, 255), "Tejido graso": (255, 255, 0), "Eritroblasto": (255, 0, 255), "Trabecula Osea": (0, 255, 255), "Granulocito": (255, 255, 255), "Region*": (0, 0, 0), "Otros": (0, 0, 0)}

    mask = Image.new('RGB', (width_img, height_img), 0)
    mask_np = np.zeros((len(possible_label), height_img, width_img), dtype=np.uint8)
    image_area = Polygon([(x_img, y_img), (x_img + width_img, y_img), (x_img + width_img, y_img + height_img), (x_img, y_img + height_img)])
    for annotation in annotations:
        # Annotation is a paquo.pathobjects.QuPathPathAnnotationObject
        ann = annotation.roi # This is a Polygon object
        class_name = annotation.path_class.name

        try:
            label = label_colour[class_name]
            int_label = possible_label[class_name]  
        except KeyError:
            raise ValueError(f"Skipping annotation {annotation.name} (unknown class {class_name})")

        if type(ann) != Polygon:
            print(f"Skipping annotation {annotation.name} (not a polygon)")
            continue
        
        # polygon is within the image region
        if image_area.intersects(ann):
            inter_area = image_area.intersection(ann)


            if isinstance(inter_area, Polygon):
                inter_area_coords = [(x - x_img, y - y_img) for x,y in inter_area.exterior.coords]
    
                ImageDraw.Draw(mask).polygon(inter_area_coords, outline=label, fill=label)
                aux_img = Image.new('L', (width_img, height_img), 0)
                ImageDraw.Draw(aux_img).polygon(inter_area_coords, outline=1, fill=1)
                mask_np[int_label] |= np.array(aux_img, dtype=np.uint8)
            
            elif isinstance(inter_area, (MultiPolygon, GeometryCollection)):
                for p in inter_area.geoms:
                    if isinstance(p, (LineString, MultiLineString)):
                        continue

                    inter_area_coords = [(x - x_img, y - y_img) for x,y in p.exterior.coords]
                    ImageDraw.Draw(mask).polygon(inter_area_coords, outline=label, fill=label)
                    aux_img = Image.new('L', (width_img, height_img), 0)
                    ImageDraw.Draw(aux_img).polygon(inter_area_coords, outline=1, fill=1)
                    mask_np[int_label] |= np.array(aux_img, dtype=np.uint8)

            elif type(inter_area) == LineString or type(inter_area) == MultiLineString:
                continue
            else:
                raise ValueError(f"Unexpected intersection type {type(inter_area)}")
            
    return mask, mask_np

def extract_images_from_svs(svs_file: str, output_dir: str, qp_img: QuPathProjectImageEntry,args: argparse.Namespace):
    slide = openslide.OpenSlide(svs_file)
    width, height = slide.dimensions
    file_name = os.path.basename(svs_file).replace(".svs", "").replace(" ", "_")
    print(f"Extracting images from {file_name}")

    stride = args.stride
    patch_size = args.patch_size

    if stride < patch_size:
        padding = (patch_size - stride) // 2
    else:
        padding = 0

    if padding > 0:
        padded_width = width + 2 * padding
        padded_height = height + 2 * padding
    else:
        padded_width, padded_height = width, height

    num_images_x = (padded_width - patch_size) // stride + 1
    num_images_y = (padded_height - patch_size) // stride + 1

    print(f"Extracting {num_images_x * num_images_y} images with patch size {patch_size}, stride {stride}, and padding {padding}")

    for i in range(num_images_x):
        for j in range(num_images_y):
            x = i * stride - padding
            y = j * stride - padding

            img = slide.read_region((max(0, x), max(0, y)), 0, (patch_size, patch_size))  

            img_array = np.array(img)
            
            # Check if the image is completely white (or close)
            if np.mean(img_array) > 240:
                #logging.debug(f"Skipping image {i}_{j} (mostly white)")
                # If the image is mostly white, skip it
                continue
            
            mask, mask_np = extract_annotations(qp_img, x, y, patch_size, patch_size)
            
            if np.mean(mask_np) == 0 and not args.no_masks:
                logging.debug(f"Skipping image {i}_{j} (no annotations)")
                # If there are no annotations, skip it
                continue

            logging.debug(f"Saving image {file_name} {i}_{j}")
            
            img_overlay = Image.blend(img.convert("RGB"), mask, alpha=0.3)
            
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("x", str(x))
            metadata.add_text("y", str(y))

            # Convert the image to 8-bit and save
            img = img.convert("RGB")
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
                
            img.save(os.path.join(output_dir, f"images/{file_name}_{i}_{j}.png"), pnginfo=metadata)

            def save_mask(mask):
                n_labels, height, width = mask.shape
                for lab in range(n_labels):
                    img = Image.fromarray(mask[lab] * 255)  
                    if np.mean(mask[lab]) == 0:
                        continue
                    img.save(os.path.join(output_dir, f"masks/img/{file_name}_{i}_{j}_label_{lab}.png"))  

            if not args.no_masks:   
                img_overlay.save(os.path.join(output_dir, f"overlay/{file_name}_{i}_{j}.png"), pnginfo=metadata)
                mask.save(os.path.join(output_dir, f"masks/img/{file_name}_{i}_{j}.png"), pnginfo=metadata)
                save_mask(mask_np)
                np.save(os.path.join(output_dir, f"masks/data/{file_name}_{i}_{j}.npy"), mask_np)


    print(f"Extracted {num_images_x * num_images_y} images")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract patches from an SVS file")
    
    parser.add_argument('--output_dir', default="./data/", type=str, help='Directory to save the extracted images')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of each patch (default: 256)')
    parser.add_argument('--stride', type=int, help='Stride between patches (default: 256)')
    parser.add_argument('--no_masks', action='store_true', help='Do not save masks')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.stride:
        args.stride = args.patch_size

    qp = QuPathProject("data/ours/raw/annotations/project.qpproj", mode='r')
    qp_img_list = {img.image_name: i for i, img in enumerate(qp.images)}
    visited_files = []

    for folder in sorted(os.listdir("./data/ours/raw/Image/196/")):
        #svs_path = os.path.join("./data/ours/raw/Image/", folder, "Binary")
        svs_path = os.path.join("./data/ours/raw/Image/196", "Binary")
        print(f"Extracting image from {svs_path}")
        for file in os.listdir(svs_path):
            svs_file = os.path.join(svs_path, file)
            if not file.endswith(".svs") or file in visited_files:
                continue
            print(f"Extracting images {file}")
            qp_img = qp.images[qp_img_list[file]]
            print(qp_img.hierarchy.annotations)
            visited_files.append(file)

            extract_images_from_svs(svs_file, args.output_dir, qp_img, args)
            
