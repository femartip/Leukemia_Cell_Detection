import json
from PIL import Image
import os
import cv2

def read_image_metadata(image_path):
    img = Image.open(image_path)
    metadata = img.info
    x = metadata.get("x")
    y = metadata.get("y")
    return x, y

img_size = (57631,43304)

geojson = {
    "type": "FeatureCollection",
    "features": []
}

json_img = {
            "type": "Feature",
            "id": "aa45a6cd-ee8b-4200-8afc-6b0b2f8ef597",
            "geometry": {
                "type":"Polygon",
                "coordinates":[]
            },
            "properties":{
                "objectType":"annotation",
                "classification":{"name":"","color":[]}
            }
}

images = []
images_names = []
image_coordinates = []
max_images = 2
n_images = 0

for file in os.listdir("./data/images/256/"):
    if file.endswith(".png") and n_images < max_images:
        images_names.append(str(file))
        image_bgr = cv2.imread(os.path.join("./data/images/256/", file))
        x,y = read_image_metadata(os.path.join("./data/images/256/", file))
        #image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        images.append(image_bgr)
        image_coordinates.append((x, y))
        n_images += 1

for i in range(len(images)):
    json__for_img = json_img.copy()
    X,Y = int(image_coordinates[i][0]), int(image_coordinates[i][1])
    json__for_img["geometry"]["coordinates"] = [[[X,Y],[X+256,Y],[X+256,Y+256],[X,Y+256,X], [X,Y]]]
    geojson["features"].append(json__for_img)

with open('data/annotations.geojson', 'w') as f:
    json.dump(geojson, f)
