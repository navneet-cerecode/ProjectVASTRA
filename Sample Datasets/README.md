# Sample Datasets

This folder contains sample images from the **official HR-VITON dataset**, used for demonstration and testing purposes.

## About the Images

- The **parsing images** provided here are **visualized versions** (colored) for easier human understanding and debugging.
- When using your **own custom dataset**, make sure to use the **grayscale parsing images** instead, as those are the ones actually used by the model during inference.

## Note on Visualized vs Grayscale Parse Maps

- **Grayscale parse maps**:  
  Used during model training and inference. These are 1-channel images where each pixel value represents a different body part class.

- **Visualized parse maps**:  
  Used for visualization and debugging only. These are 3-channel RGB images created by mapping labels to specific colors.

## Custom Dataset Reminder

When generating your own dataset:
- Use the grayscale `.png` files for parsing.
- The visualized versions (if created) should not be fed into the model.
- Ensure the naming conventions are consistent with HR-VITON standards.

---

> ⚠️ **Disclaimer:** The sample images in this folder are from the original HR-VITON dataset and are subject to the same license restrictions. They are provided here for non-commercial, educational, and research purposes only.

