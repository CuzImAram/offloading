import sys
import numpy as np
from PIL import Image
import os

def calculate_difference(path1, path2):
    try:
        # Load images and convert to RGB (standardizes PNG vs JPG)
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')
    except Exception as e:
        print(f"Error: {e}")
        return

    # If sizes differ, resize the second image to match the first
    if img1.size != img2.size:
        print(f"Note: Dimensions differ ({img1.size} vs {img2.size}). Resizing for comparison...")
        img2 = img2.resize(img1.size)

    # Convert images to numpy arrays
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)

    # Calculate the Mean Absolute Error (MAE)
    # This finds the average difference between pixels across all channels (R, G, B)
    diff = np.abs(arr1 - arr2)
    mean_diff = np.mean(diff)

    # Scale to a percentage (255 is the max possible difference for 8-bit pixels)
    percentage = (mean_diff / 255.0) * 100

    # Create difference image with red markers
    pixel_diff = np.any(diff > 0, axis=2)
    diff_image = arr1.copy().astype(np.uint8)
    diff_image[pixel_diff] = [255, 0, 0]  # Mark different pixels in red
    
    # Save difference image
    base_name = os.path.splitext(os.path.basename(path1))[0]
    diff_path = f"{base_name}_diff.jpg"
    Image.fromarray(diff_image).save(f"images/{diff_path}")

    if percentage == 0:
        print("The images are identical (0% difference).")
    else:
        print(f"The images are different.")
        print(f"Difference Gauge: {percentage:.2f}%")
        print(f"Difference image saved to: images/{diff_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <image1> <image2>")
    else:
        calculate_difference(sys.argv[1], sys.argv[2])
