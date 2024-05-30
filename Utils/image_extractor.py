import openslide
import os
import numpy as np
from PIL import Image, PngImagePlugin

#Extract patches from a svs file

def extract_images_from_svs(svs_file, output_dir):
    # Open the svs file
    slide = openslide.OpenSlide(svs_file)

    # Get the dimensions of the slide
    width, height = slide.dimensions

    # Define the size of each extracted image
    patch_size = 256

    # Calculate the number of images in each dimension
    num_images_x = width // patch_size
    num_images_y = height // patch_size

    print(f"Extracting {num_images_x * num_images_y} images")
    
    # Loop over the slide to extract images
    for i in range(num_images_x):
        for j in range(num_images_y):
            # Calculate the position of the current image
            x = i * patch_size
            y = j * patch_size

            # Extract the image
            img = slide.read_region((x, y), 0, (patch_size, patch_size))

            # Convert the image to a numpy array
            img_array = np.array(img)
            #print(np.mean(img_array))
            # Check if the image is completely white
            if np.mean(img_array) > 240:
                print(f"Skipping image {i}_{j}")
                # If the image is completely white, skip it
                continue
            
            print(f"Saving image {i}_{j}")

            # Create metadata object
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("x", str(x))
            metadata.add_text("y", str(y))

            # Convert the image to 8-bit
            img = img.convert("RGB")
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)

            # Save the image
            img.save(os.path.join(output_dir, f"image_{i}_{j}.png"), pnginfo=metadata)

    print(f"Extracted {num_images_x * num_images_y} images")
    
output_dir = "./data/images/256_all"
#output_dir = "./data/256"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
svs_file = "./data/Caso_1.svs"
extract_images_from_svs(svs_file, output_dir)
