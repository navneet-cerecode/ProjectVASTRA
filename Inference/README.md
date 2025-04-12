
---

# Project VASTRA

**Project VASTRA** (Virtual Apparel Simulation for Try-on and Realistic Augmentation) is a high-resolution virtual try-on system built using the [HR-VITON](https://arxiv.org/abs/2206.14180) framework. It enables users to visualize how clothes might look when worn by a person, using deep learning models designed to handle misalignment and occlusion in human images.This project is not for commercial purposes, as the pretrained models used in

> Based on the paper:  
> **High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions**  
> [arXiv:2206.14180](https://arxiv.org/abs/2206.14180)  
> GitHub Repo: [sangyun884/HR-VITON](https://github.com/sangyun884/HR-VITON?tab=readme-ov-file)

---

## Features

- High-resolution (1024×768) try-on outputs
- Occlusion-aware clothing generation
- Uses parsed human parts, dense pose, and pose keypoints
- Supports custom datasets

---

## Installation & Setup

### 1. Create Environment

```bash
conda create -n vastra_env python=3.8
conda activate vastra_env
```

### 2. Install Dependencies

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install opencv-python torchgeometry Pillow tqdm tensorboardX scikit-image scipy
```

---

## Preprocessing Guide

> ⚠️ Follow the complete preprocessing guide located in the [`data`](./data) folder's README before running inference.

Make sure you resize all image folders and generate `test_pairs.txt` using the provided scripts.

### `test_pairs.txt` Generation Script

Use the Python script to automatically match person and cloth image pairs for testing.

### Image Resizing Script

Use the provided script to resize all image-related folders as required by the model.

> All directory paths in the preprocessing scripts are **hardcoded**, so make sure to **update them as per your setup**.

---

## 📥 Pretrained Models

Please download the following pretrained weights and place them in:

```
./eval_models/weights/v0.1/
```

| Component                     | Link                                                                                      |
|------------------------------|-------------------------------------------------------------------------------------------|
| Try-on Condition Generator   | [Download](https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ/view)       |
| Try-on Condition Discriminator | [Download](https://drive.google.com/file/d/1T4V3cyRlY5sHVK7Quh_EJY5dovb5FxGX/view)     |
| Try-on Image Generator       | [Download](https://drive.google.com/file/d/1T4V3cyRlY5sHVK7Quh_EJY5dovb5FxGX/view)       |
| AlexNet (LPIPS metric)       | [Download](https://drive.google.com/file/d/1FF3BBSDIA3uavmAiuMH6YFCv09Lt8jUr/view)       |

---

## 🚀 Running Inference

Make sure that both the person image, cloth and their preprocessed outputs should have same name.

After completing preprocessing and downloading models:

```bash
python -u test_generator.py \
  --occlusion \
  --cuda True \
  --test_name "testing1" \
  --gpu_ids "0" \
  --tocg_checkpoint "eval_models/weights/v0.1/mtviton.pth" \
  --gen_checkpoint "eval_models/weights/v0.1/gen.pth" \
  --datasetting "unpaired" \
  --dataroot "data" \
  --data_list "test_pairs.txt"
```

---

## Citation

If you use this project or the original HR-VITON model, please cite the following paper:

```bibtex
@article{lee2022hrviton,
  title={High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions},
  author={Lee, Sangyun and Gu, Gyojung and Park, Sunghyun and Choi, Seunghwan and Choo, Jaegul},
  journal={arXiv preprint arXiv:2206.14180},
  year={2022}
}
```

---

## Acknowledgements

This project uses the publicly released models and code from:

- [HR-VITON GitHub](https://github.com/sangyun884/HR-VITON)
- [HR-VITON Research Paper](https://arxiv.org/abs/2206.14180)

---

## License and Disclaimer

This project integrates **pre-trained models and core components** from the [HR-VITON repository](https://github.com/sangyun884/HR-VITON), developed by Sangyun Park *et al.*.

> **License Notice:**  
> The HR-VITON project is distributed under a **custom academic license** that explicitly prohibits **commercial use**.

As such, this derived work:

- Is intended **solely for academic, research, and demonstration purposes**.
- **May not be used for any commercial applications**.
- Fully credits the original authors of HR-VITON for their contributions.

Please review the original [HR-VITON License File](https://github.com/sangyun884/HR-VITON/blob/master/LICENSE) for detailed legal terms.

---



