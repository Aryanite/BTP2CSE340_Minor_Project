"""
TransUNet-RS — FastAPI Backend
================================
REST API for satellite image segmentation.

Endpoints:
  POST /predict       — Upload an image, receive the segmentation map.
  GET  /health        — Health check.
  GET  /classes       — List supported LULC classes.
  GET  /              — API information.

Usage::

    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import base64
from typing import Dict, List

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.inference.predict import Predictor
from src.dataset.data_loader import EUROSAT_CLASSES


# ===================================================================== #
#  App Configuration
# ===================================================================== #
CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG", "configs/model_config.yaml")
DEVICE = os.getenv("DEVICE", "cuda")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

# ===================================================================== #
#  FastAPI App
# ===================================================================== #
app = FastAPI(
    title="TransUNet-RS API",
    description=(
        "Satellite image LULC segmentation powered by a hybrid "
        "CNN-Transformer architecture."
    ),
    version="1.0.0",
)

# CORS — allow frontend to call from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global predictor (lazy-loaded) ───────────────────────────────────
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Lazy-load the model predictor on first request."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor(
            checkpoint_path=CHECKPOINT_PATH,
            model_config_path=MODEL_CONFIG_PATH,
            device=DEVICE,
            image_size=IMAGE_SIZE,
        )
    return _predictor


# ===================================================================== #
#  Routes
# ===================================================================== #
@app.get("/", tags=["Info"])
async def root() -> Dict:
    """API root — basic information."""
    return {
        "service": "TransUNet-RS Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST — Upload image for segmentation",
            "/health": "GET — Health check",
            "/classes": "GET — List LULC classes",
        },
    }


@app.get("/health", tags=["Info"])
async def health() -> Dict:
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": _predictor is not None}


@app.get("/classes", tags=["Info"])
async def list_classes() -> Dict[str, List]:
    """Return the list of supported LULC classes."""
    return {
        "num_classes": len(EUROSAT_CLASSES),
        "classes": [
            {"index": i, "name": name}
            for i, name in enumerate(EUROSAT_CLASSES)
        ],
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a satellite image and receive the segmentation result.

    Parameters
    ----------
    file : UploadFile
        Image file (TIFF, PNG, JPEG).

    Returns
    -------
    JSON with:
      - ``segmentation_map``: base64-encoded colorized PNG.
      - ``class_distribution``: per-class pixel percentages.
      - ``dominant_class``: name of the most prevalent class.
    """
    # Validate file type
    allowed_types = {"image/tiff", "image/png", "image/jpeg", "image/jpg"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {allowed_types}",
        )

    # Read file
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_MB} MB.",
        )

    # Decode image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run prediction
    predictor = get_predictor()
    pred_mask, pred_rgb, prob_map = predictor.predict(image_np)

    # Encode colorized prediction to base64 PNG
    pred_image = Image.fromarray(pred_rgb)
    buffer = io.BytesIO()
    pred_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Compute class distribution
    total_pixels = pred_mask.size
    distribution = {}
    for idx, name in enumerate(EUROSAT_CLASSES):
        count = int((pred_mask == idx).sum())
        distribution[name] = round(count / total_pixels * 100, 2)

    # Dominant class
    dominant_idx = int(np.bincount(pred_mask.flatten(), minlength=len(EUROSAT_CLASSES)).argmax())
    dominant_class = EUROSAT_CLASSES[dominant_idx]

    return JSONResponse(
        content={
            "segmentation_map": img_base64,
            "class_distribution": distribution,
            "dominant_class": dominant_class,
            "image_size": {
                "width": image_np.shape[1],
                "height": image_np.shape[0],
            },
        }
    )


@app.post("/predict/image", tags=["Inference"])
async def predict_image(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a satellite image and receive the colorized segmentation
    map directly as a PNG image response.

    Parameters
    ----------
    file : UploadFile
        Image file.

    Returns
    -------
    StreamingResponse — PNG image.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    predictor = get_predictor()
    _, pred_rgb, _ = predictor.predict(image_np)

    pred_image = Image.fromarray(pred_rgb)
    buffer = io.BytesIO()
    pred_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# ===================================================================== #
#  Entry point
# ===================================================================== #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.inference.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
