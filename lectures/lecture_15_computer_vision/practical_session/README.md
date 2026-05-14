# Lecture 15 Practical Session

This directory contains the student practical for an introductory Computer Vision lecture.

## Files

| File | Description |
| --- | --- |
| `computer_vision_practical_student_90min.ipynb` | Student notebook with guided TODO cells and runnable visual demos |
| `computer_vision_practical_student_90min.py` | Generated companion script for review and diffing |

## Structure

The practical follows a visual storyline:

| Part | Topic | Dataset / assets |
| --- | --- | --- |
| 1 | Images as tensors and historical note | Lenna, `skimage.data` |
| 2 | Classical convolution filters | `camera`, `astronaut` |
| 3 | Tiny CNN training | Fashion-MNIST |
| 4 | Detection, segmentation, and YOLO | TorchVision and Ultralytics demos |
| 5 | Image embeddings and similarity search | ResNet-18, CIFAR-10 sample |
| 6 | Multimodal embeddings bridge | CLIP / OpenCLIP demo |
| 7 | Modern CV snapshot | SAM 2, YOLO26, RF-DETR, DINOv3, SigLIP 2, Florence-2 |

## Environment

Google Colab: the notebook auto-installs missing lightweight dependencies.

Local:

```bash
uv sync --group neural_networks
uv run python -m pip install scikit-image
```

The extra `scikit-image` install is only needed while this draft lecture is not yet promoted
to a full dependency group in `pyproject.toml`.

## Notes

- Live demos are enabled by default: YOLO, TorchVision detection, TorchVision segmentation, and CLIP/OpenCLIP.
- These cells may install extra packages and download model weights; turn the `RUN_*_DEMO` flags off for a shorter or offline class.
- Lenna is included only as a short historical note; the main practical uses neutral sample images.
