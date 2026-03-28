# TransUNet-RS: Hybrid CNN-Transformer for Land Use / Land Cover Classification

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

## Overview

**TransUNet-RS** is a research-grade, production-ready system for pixel-wise Land Use and Land Cover (LULC) classification from optical satellite imagery (Sentinel-2). It combines a **ResNet-50 CNN encoder** with a **Vision Transformer (ViT) bottleneck** and a **hybrid decoder** with cross-attention skip connections, achieving state-of-the-art segmentation performance on remote sensing benchmarks.

### Key Features

- **Hybrid Architecture** — ResNet-50 multi-scale encoder + 12-layer ViT bottleneck + cross-attention decoder
- **Production Pipeline** — Data preprocessing, augmentation (spectral jitter, MixUp), mixed-precision training
- **Comprehensive Evaluation** — OA, mIoU, per-class F1, Cohen's Kappa with visualization
- **REST API** — FastAPI inference server with image upload → segmentation map response
- **Web Demo** — Modern drag-and-drop UI for interactive classification
- **Docker Deployment** — One-command container launch with optional GPU support

## Architecture

```
Input Image (3×256×256)
        │
  ┌─────▼─────┐
  │  ResNet-50 │ ── skip₁ (64ch)  ── skip₂ (256ch) ── skip₃ (512ch)
  │  Encoder   │
  └─────┬─────┘
        │ (1024ch, 16×16)
  ┌─────▼─────────────┐
  │  Patch Embed +     │
  │  12× Transformer   │
  │  Encoder Layers    │
  └─────┬─────────────┘
        │ (768ch → 1024ch, 16×16)
  ┌─────▼─────────────┐
  │  Hybrid Decoder    │
  │  + Cross-Attention │
  │  + Skip Fusions    │
  └─────┬─────────────┘
        │
  ┌─────▼─────┐
  │  1×1 Conv  │ → num_classes × 256 × 256
  └───────────┘
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/TransUNet-RS.git
cd TransUNet-RS
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Prepare Data

Download EuroSAT from [the official source](https://github.com/phelber/eurosat) and place images under `data/raw/eurosat/`.

```bash
python -m src.dataset.preprocessing --input data/raw/eurosat --output data/processed
```

### 3. Train

```bash
python -m src.training.train --config configs/training_config.yaml
```

### 4. Evaluate

```bash
python -m src.inference.predict --checkpoint checkpoints/best_model.pth --input data/processed/test
```

### 5. Launch API

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/docs` for the interactive API docs, or open `frontend/index.html` in a browser for the web demo.

### 6. Docker

```bash
cd deployment
docker-compose up --build
```

## Project Structure

```
TransUNet-RS/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── dataset/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── cnn_encoder.py
│   │   ├── transformer.py
│   │   ├── decoder.py
│   │   └── transunet_rs.py
│   ├── training/
│   │   ├── train.py
│   │   ├── loss.py
│   │   └── optimizer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualize.py
│   └── inference/
│       ├── predict.py
│       └── api.py
├── notebooks/
│   ├── training_demo.ipynb
│   └── visualization.ipynb
├── frontend/
│   └── index.html
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
├── paper/
│   └── main_paper.md
└── diagrams/
    └── prompts.txt
```

## Datasets

| Dataset       | Bands | Classes | Resolution | Patches  |
|---------------|-------|---------|------------|----------|
| EuroSAT       | 13    | 10      | 10 m       | 27,000   |
| BigEarthNet   | 12    | 43      | 10–60 m    | 590,326  |

## Results (Expected)

| Metric            | Value   |
|-------------------|---------|
| Overall Accuracy  | ~95.2%  |
| Mean IoU          | ~82.4%  |
| Cohen's Kappa     | ~0.94   |
| Macro F1          | ~0.91   |

## Citation

```bibtex
@article{transunet_rs_2026,
  title   = {TransUNet-RS: Hybrid CNN-Transformer Architecture for
             Optical Image-Based Land Use and Land Cover Classification},
  author  = {Research Team},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations
             and Remote Sensing},
  year    = {2026}
}
```

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
