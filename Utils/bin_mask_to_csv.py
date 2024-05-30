# Read all the file names in data/bin_masks and write them to masks.csv

import os
import pandas as pd

def index(file):
    return int(file.split("_")[1].split(".")[0])

masks = sorted(os.listdir("./data/bin_masks/"), key=index)
masks_df = pd.DataFrame(masks, columns=["label"])
masks_df.to_csv("./data/masks.csv", index=False)
print("Done")