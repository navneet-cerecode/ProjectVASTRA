import os
from PIL import Image
from tqdm import tqdm

# Root directory of your dataset
root_dir = r'C:\Users\nikun\OneDrive\Desktop\ProjectVASTRA\Inference\data\test'

# Folder resize configuration: (target size (W, H), interpolation mode)
resize_map = {
    'image': ((768, 1024), Image.BICUBIC),
    'cloth': ((768, 1024), Image.BICUBIC),
    'cloth-mask': ((768, 1024), Image.NEAREST),
    'agnostic-v3.2': ((768, 1024), Image.BICUBIC),
    'image-densepose': ((768, 1024), Image.BICUBIC),
    'image-parse-agnostic-v3.2': ((768, 1024), Image.NEAREST),
    'image-parse-v3': ((768, 1024), Image.NEAREST),
    'openpose_img': ((768, 1024), Image.BICUBIC),
    # 'openpose_json' is skipped (not an image folder)
}

def resize_folder(folder_name, size, interp):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return

    for file in tqdm(os.listdir(folder_path), desc=f"Resizing {folder_name}/"):
        file_path = os.path.join(folder_path, file)
        try:
            with Image.open(file_path) as img:
                img_resized = img.resize(size, interp)
                img_resized.save(file_path)
        except Exception as e:
            print(f"❌ Error resizing {file_path}: {e}")

def main():
    print(f"\n📁 Resizing images in: {root_dir}\n")
    for folder, (size, interp) in resize_map.items():
        resize_folder(folder, size, interp)
    print("\n✅ All images resized in-place successfully!")

if __name__ == "__main__":
    main()
