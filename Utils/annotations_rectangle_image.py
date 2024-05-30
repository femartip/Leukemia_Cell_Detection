from PIL import Image
import os
import cv2
import numpy as np
from copy import deepcopy
import json

def read_image_metadata(image_path):
    img = Image.open(image_path)
    metadata = img.info
    x = metadata.get("x")
    y = metadata.get("y")
    return int(x), int(y)

geojson_template = {
            "type": "Feature",
            "id": "aa45a6cd-ee8b-4200-8afc-6b0b2f8ef597-",
            "geometry": {
                "type":"Polygon",
                "coordinates":[]
            },
            "properties":{
                "objectType":"annotation"
            }
}

geojson = {"type": "FeatureCollection",
    "features": []
}


print("Reading images")
#Read
for file in os.listdir("./data/images/256/"):
    if file.endswith(".png"):
        image_bgr = cv2.imread(os.path.join("./data/images/256/", file))
        x,y = read_image_metadata(os.path.join("./data/images/256/", file))
        geojson_rectangle = deepcopy(geojson_template)
        geojson_rectangle["geometry"]["coordinates"] = [[[x,y],[x+256,y],[x+256,y+256],[x,y+256],[x,y]]]
        geojson["features"].append(geojson_rectangle)

print("Saving geojson")
with open("./data/annotations/rectangel_annotations.geojson", "w") as f:
    f.write(json.dumps(geojson, indent=4))



