# CUDA-Accelerated Infrastructure Crack Detection

A three-stage pipeline for detecting structural cracks in infrastructure using transfer learning and GPU-accelerated image processing. Trained on 126,092 images across three real-world datasets and deployed live on Hugging Face Spaces.

## Overview

This project investigates whether pretrained ImageNet architectures can resolve domain gaps across multiple crack detection datasets without dataset-specific tuning. It combines a classification pipeline with GPU-accelerated edge detection for visual analysis.

## Results

| Model | Accuracy | Dataset |
|---|---|---|
| Custom CNN | 99.29% | SDNET2018 + CIFAR-10 |
| EfficientNetB0 | 98.14% | 3-source pooled dataset (126,092 images) |

EfficientNetB0 transfer learning outperformed the custom CNN baseline by 9.18 percentage points on pooled multi-source data, demonstrating that pretrained feature reuse helps resolve cross-dataset domain gap.

## GPU Acceleration

GPU vs CPU benchmarking was run on a Tesla T4 for Canny and Sobel edge detection.

- At 512x512, PCIe transfer overhead dominates and CPU can outperform GPU.
- At 4096x4096, GPU acceleration achieves up to 1.70x speedup on Sobel edge detection.

## Live Demo

Deployed at: [huggingface.co/spaces/Ab0y04/crack-detection-v3](https://huggingface.co/spaces/Ab0y04/crack-detection-v3)

Upload an image of infrastructure (wall, road, bridge surface) and receive a crack classification along with GPU-accelerated edge visualization.

## Tech Stack

Python, TensorFlow, CuPy, OpenCV, NumPy, Matplotlib, Hugging Face Spaces

## Key Notes

- EfficientNetB0 requires raw pixel inputs in the [0, 255] range with its built-in preprocessing layer. Applying an additional `rescale=1/255` causes double normalization and collapses accuracy to roughly 65%.
- The custom CNN baseline performs well on a single dataset but degrades significantly when evaluated on pooled multi-source data, motivating the move to transfer learning.

## License

MIT License
