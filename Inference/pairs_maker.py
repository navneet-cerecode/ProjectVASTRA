import os
import pandas as pd

# Define paths
test_dir = r"C:\Users\nikun\OneDrive\Desktop\ProjectVASTRA\Inference\data\test"
output_file = r"C:\Users\nikun\OneDrive\Desktop\ProjectVASTRA\Inference\data\test_pairs.txt"

# Get list of images and clothes
image_dir = os.path.join(test_dir, "image")
cloth_dir = os.path.join(test_dir, "cloth")

# Check if directories exist
if not os.path.exists(image_dir) or not os.path.exists(cloth_dir):
    raise FileNotFoundError(f"One of the required folders is missing: {image_dir} or {cloth_dir}")

# Get valid filenames
def get_valid_filenames(directory):
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

images = get_valid_filenames(image_dir)
clothes = get_valid_filenames(cloth_dir)

# Ensure we have at least one image and one cloth
if not images or not clothes:
    raise ValueError("No images or clothes found in the respective directories.")

# Ensure the number of images matches the number of clothes
min_length = min(len(images), len(clothes))

# Pair each image with only ONE cloth
pairs = list(zip(images[:min_length], clothes[:min_length]))

# Save to file
with open(output_file, "w") as f:
    for image, cloth in pairs:
        f.write(f"{image} {cloth}\n")

print(f"✅ test_pairs.txt saved at: {output_file}")
