import json


goejson_path = "./data/annotations_all.geojson"
with open(goejson_path) as f:
    data = json.load(f)

new_geojson = {"type": "FeatureCollection",
    "features": []
}
for feature in data["features"]:
    if len(feature["geometry"]["coordinates"][0]) < 3:
        print(f"Feature {feature['id']} has less than 3 coordinates")
    else:
        new_geojson["features"].append(feature)

out_file = open('data/annotations_new_all.geojson', 'w')
json.dump(new_geojson, out_file, indent=2)

