import os
import glob
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

# Load RMBG pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=0)
print("✅ RMBG Pipeline loaded (using CPU)")

# --- Process single image (JPG/PNG) ---
image_path = r"C:\Users\nikun\OneDrive\Desktop\HR_VITON Preprocessing\ClothMaskingAgain\input\yo.jpg"
image = Image.open(image_path).convert("RGB")

# Get mask and masked image
pillow_mask = pipe(image, return_mask=True)
pillow_image = pipe(image)

# Save results
out_dir = r"C:\Users\nikun\OneDrive\Desktop\HR_VITON Preprocessing\ClothMaskingAgain\output"
os.makedirs(out_dir, exist_ok=True)

Image.fromarray(np.array(pillow_mask)).save(os.path.join(out_dir, "yo.png"))
pillow_image.save(os.path.join(out_dir, "test_masked.png"))

print("✅ Single image processed and saved.")

# --- Optional batch processing for HR-VITON if exists ---
data_dir = r"C:\Users\nikun\OneDrive\Desktop\HR_VITON Preprocessing\HR-VITON"
train_cloth_dir = os.path.join(data_dir, "train", "cloth")
test_cloth_dir = os.path.join(data_dir, "test", "cloth")

if os.path.exists(train_cloth_dir) and os.path.exists(test_cloth_dir):
    train_cloth_paths = glob.glob(os.path.join(train_cloth_dir, "*.jpg")) + glob.glob(os.path.join(train_cloth_dir, "*.png"))
    test_cloth_paths = glob.glob(os.path.join(test_cloth_dir, "*.jpg")) + glob.glob(os.path.join(test_cloth_dir, "*.png"))

    os.makedirs("./mask/train", exist_ok=True)
    os.makedirs("./mask/test", exist_ok=True)

    for path in tqdm(train_cloth_paths, desc="Train masks"):
        name = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        mask = pipe(img, return_mask=True)
        mask_np = np.array(mask.convert("L"))
        inverted_mask = 255 - mask_np
        plt.imsave(os.path.join("./mask/train", name), inverted_mask, cmap="gray")

    for path in tqdm(test_cloth_paths, desc="Test masks"):
        name = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        mask = pipe(img, return_mask=True)
        mask_np = np.array(mask.convert("L"))
        inverted_mask = 255 - mask_np
        plt.imsave(os.path.join("./mask/test", name), inverted_mask, cmap="gray")

    print("✅ Batch processing done.")
else:
    print("⚠ VASTRA folders not found. Skipping batch.")
