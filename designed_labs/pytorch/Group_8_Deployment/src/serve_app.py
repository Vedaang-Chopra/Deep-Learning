"""
serve_app.py — Minimal REST Inference Server (FastAPI)
========================================================

Student implements:
  - /health endpoint
  - /predict endpoint (single image)
  - /predict_batch endpoint (batch of images)
  - Integration with Batcher for dynamic batching
  - Request/response logging

No HuggingFace serving. No Triton/vLLM. Manual FastAPI server.
"""

# ╔═══════════════════════════════════════════════════════╗
# ║  NOTE: This file defines a FastAPI application.        ║
# ║  Run with: uvicorn serve_app:app --host 0.0.0.0       ║
# ║                                                       ║
# ║  Install: pip install fastapi uvicorn python-multipart ║
# ╚═══════════════════════════════════════════════════════╝

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import time
import io
import base64
import logging
from typing import List, Optional

# FastAPI imports — wrap in try/except for environments without it
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ FastAPI not installed. Install with: pip install fastapi uvicorn")


# ─────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class PredictRequest(BaseModel):
        """Single prediction request."""
        image_b64: str  # base64-encoded image bytes
        return_probs: bool = False

    class PredictResponse(BaseModel):
        """Single prediction response."""
        class_id: int
        class_name: str
        confidence: float
        latency_ms: float
        probs: Optional[List[float]] = None

    class BatchPredictRequest(BaseModel):
        """Batch prediction request."""
        images_b64: List[str]
        return_probs: bool = False

    class BatchPredictResponse(BaseModel):
        """Batch prediction response."""
        predictions: List[PredictResponse]
        batch_size: int
        total_latency_ms: float


# ─────────────────────────────────────────────────────
# CIFAR-10 Classes (provided)
# ─────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ─────────────────────────────────────────────────────
# Server Application
# ─────────────────────────────────────────────────────

def create_app(
    model: nn.Module = None,
    model_path: str = None,
    device: str = "cpu",
    use_batcher: bool = False,
    max_batch_size: int = 32,
    max_wait_ms: float = 50.0,
):
    """
    Create a FastAPI inference application.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Create FastAPI() app                               ║
    ║  2. Load or use provided model                         ║
    ║  3. Set up preprocessing transform                     ║
    ║  4. Optionally create Batcher                          ║
    ║  5. Implement endpoints (below)                        ║
    ║  6. Return app                                         ║
    ╚═══════════════════════════════════════════════════════╝
    """

    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed")

    app = FastAPI(title="CIFAR-10 Inference Server")
    logger = logging.getLogger("inference_server")

    # Preprocessing
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # ── /health ──────────────────────────────────────

    @app.get("/health")
    def health():
        """
        Health check endpoint.

        ╔═══════════════════════════════════════════════════╗
        ║  TODO: Return status, device, model info.         ║
        ║                                                   ║
        ║  Return: {"status": "healthy",                    ║
        ║           "device": str(device),                  ║
        ║           "model_loaded": model is not None}      ║
        ╚═══════════════════════════════════════════════════╝
        """
        raise NotImplementedError("TODO: implement /health")

    # ── /predict ─────────────────────────────────────

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        """
        Single image prediction.

        ╔═══════════════════════════════════════════════════╗
        ║  TODO: Implement.                                 ║
        ║                                                   ║
        ║  1. Decode base64 image → PIL Image               ║
        ║  2. Apply transform                               ║
        ║  3. Add batch dim, move to device                 ║
        ║  4. t0 = time.perf_counter()                      ║
        ║  5. Forward pass (no_grad)                        ║
        ║  6. Get predicted class + confidence              ║
        ║  7. latency = (time.perf_counter() - t0) * 1000   ║
        ║  8. Log: class, confidence, latency               ║
        ║  9. Return PredictResponse                        ║
        ╚═══════════════════════════════════════════════════╝
        """
        raise NotImplementedError("TODO: implement /predict")

    # ── /predict_batch ───────────────────────────────

    @app.post("/predict_batch", response_model=BatchPredictResponse)
    def predict_batch(request: BatchPredictRequest):
        """
        Batch prediction endpoint.

        ╔═══════════════════════════════════════════════════╗
        ║  TODO: Implement.                                 ║
        ║                                                   ║
        ║  1. Decode all base64 images                      ║
        ║  2. Stack into batch tensor                       ║
        ║  3. Forward pass on batch                         ║
        ║  4. Split results per image                       ║
        ║  5. Return BatchPredictResponse                   ║
        ║                                                   ║
        ║  Optional: Route through Batcher if use_batcher   ║
        ╚═══════════════════════════════════════════════════╝
        """
        raise NotImplementedError("TODO: implement /predict_batch")

    return app


# ─────────────────────────────────────────────────────
# Load Test Helper (provided)
# ─────────────────────────────────────────────────────

def generate_test_payload(num_images: int = 1) -> dict:
    """Generate test payload with random CIFAR-10-sized images. Provided."""
    images = []
    for _ in range(num_images):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        images.append(base64.b64encode(buf.getvalue()).decode())

    if num_images == 1:
        return {"image_b64": images[0]}
    return {"images_b64": images}


LAUNCH_INSTRUCTIONS = """
╔═══════════════════════════════════════════════════════╗
║  Launch the inference server:                         ║
║                                                       ║
║  pip install fastapi uvicorn python-multipart pillow   ║
║                                                       ║
║  # From the src/ directory:                           ║
║  uvicorn serve_app:app --host 0.0.0.0 --port 8000     ║
║                                                       ║
║  # Test:                                              ║
║  curl http://localhost:8000/health                     ║
║                                                       ║
║  # Load test (from notebook):                         ║
║  import requests                                      ║
║  payload = generate_test_payload(1)                   ║
║  resp = requests.post("http://localhost:8000/predict", ║
║                       json=payload)                    ║
║  print(resp.json())                                    ║
╚═══════════════════════════════════════════════════════╝
"""
