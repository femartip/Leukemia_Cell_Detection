import openslide
import os
import numpy as np
from PIL import Image, PngImagePlugin, ImageOps
import argparse

# Extract patches from an SVS file with padding and stride
def extract_images_from_svs(svs_file, output_dir, patch_size, stride):
    # Open the svs file
    slide = openslide.OpenSlide(    )

    # Get the dimensions of the slide
    width, height = slide.dimensions

    # Automatically calculate padding based on stride
    if stride < patch_size:
        padding = (patch_size - stride) // 2
    else:
        padding = 0

    # Apply padding to the slide if required
    if padding > 0:
        padded_width = width + 2 * padding
        padded_height = height + 2 * padding
    else:
        padded_width, padded_height = width, height

    # Calculate the number of images to extract in each dimension with stride and padding
    num_images_x = (padded_width - patch_size) // stride + 1
    num_images_y = (padded_height - patch_size) // stride + 1

    print(f"Extracting {num_images_x * num_images_y} images with patch size {patch_size}, stride {stride}, and padding {padding}")

    # Loop over the slide to extract images
    for i in range(num_images_x):
        for j in range(num_images_y):
            # Calculate the position of the current image, considering padding
            x = i * stride - padding
            y = j * stride - padding

            # Read the image region from the slide
            img = slide.read_region((max(0, x), max(0, y)), 0, (patch_size, patch_size))

            # If padding goes beyond the original image, fill the extra area
            if padding > 0:
                img = ImageOps.expand(img, border=(max(0, -x), max(0, -y), max(0, x + patch_size - width), max(0, y + patch_size - height)), fill='white')

            # Convert the image to a numpy array
            img_array = np.array(img)
            
            # Check if the image is completely white (or close)
            if np.mean(img_array) > 240:
                print(f"Skipping image {i}_{j} (mostly white)")
                # If the image is mostly white, skip it
                continue
            
            print(f"Saving image {i}_{j}")

            # Create metadata object
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("x", str(x))
            metadata.add_text("y", str(y))

            # Convert the image to 8-bit and save
            img = img.convert("RGB")
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            img.save(os.path.join(output_dir, f"image_{i}_{j}.png"), pnginfo=metadata)

    print(f"Extracted {num_images_x * num_images_y} images")

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Extract patches from an SVS file")
    
    # Arguments for image path, patch size, stride, and output directory
    parser.add_argument('svs_file', type=str, help='Path to the input SVS file')
    parser.add_argument('output_dir', type=str, help='Directory to save the extracted images')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of each patch (default: 256)')
    parser.add_argument('--stride', type=int, default=256, help='Stride between patches (default: 256)')
    
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Extract images from the SVS file
    extract_images_from_svs(args.svs_file, args.output_dir, args.patch_size, args.stride)
