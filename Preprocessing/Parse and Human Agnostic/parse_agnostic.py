import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from tqdm import tqdm

def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    """Generate agnostic parsing map by masking arms, torso, and neck."""

    # Resize parsing map to fixed size
    im_parse_resized = im_parse.resize((w, h), Image.NEAREST)
    label_array = np.array(im_parse_resized)

    # Mask upper body and neck
    parse_upper = ((label_array == 5).astype(np.float32) +
                   (label_array == 6).astype(np.float32) +
                   (label_array == 7).astype(np.float32))
    parse_neck = (label_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse_resized.copy()

    # Mask arms using pose keypoints
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]

        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or \
               (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue

            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)

            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i

        mask_arm_resized = mask_arm.resize((w, h), Image.NEAREST)
        parse_arm = (np.array(mask_arm_resized) / 255) * (label_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # Mask torso and neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic


if __name__ == "__main__":
    data_path = './test'
    output_path = './test/image-parse-agnostic-v3.2'

    os.makedirs(output_path, exist_ok=True)

    # Supported image extensions
    valid_ext = ('.jpg', '.jpeg', '.png')

    images = [f for f in os.listdir(osp.join(data_path, 'image')) if f.endswith(valid_ext)]

    for im_name in tqdm(images, desc="Processing Images", unit="image"):
        base_name = osp.splitext(im_name)[0]

        # Load OpenPose JSON
        pose_name = f"{base_name}_keypoints.json"
        pose_path = osp.join(data_path, 'openpose_json', pose_name)

        if not osp.exists(pose_path):
            print(f"❌ Missing JSON: {pose_name}")
            continue

        try:
            with open(pose_path, 'r') as f:
                pose_label = json.load(f)
                if not pose_label['people']:
                    print(f"⚠️ No people detected in {pose_name}")
                    continue
                pose_data = np.array(pose_label['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:, :2]
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            print(f"⚠️ Error loading pose data for {pose_name}: {e}")
            continue

        # Load Parsing Image
        parse_name = f"{base_name}.png"
        parse_path = osp.join(data_path, 'image-parse-v3', parse_name)

        if not osp.exists(parse_path):
            print(f"❌ Missing parsing image: {parse_name}")
            continue

        try:
            im_parse = Image.open(parse_path)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"⚠️ Error loading image: {parse_name}: {e}")
            continue

        # Generate agnostic parsing map
        agnostic = get_im_parse_agnostic(im_parse, pose_data)

        # Resize agnostic back to original input size
        if agnostic.size != im_parse.size:
            agnostic = agnostic.resize(im_parse.size, Image.NEAREST)

        # Save output
        agnostic.save(osp.join(output_path, parse_name))

    print("✅ Processing complete!")
