"""
app.py — Root entry point for Hugging Face Spaces (Gradio SDK - Free Tier).
Mounts the Flask REST API at /api/* alongside a minimal Gradio status page at /.
"""

import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # Load torch before numpy/cv2
import gradio as gr
import uvicorn
from fastapi import FastAPI
from a2wsgi import WSGIMiddleware

from backend.app import create_app

# Register a @spaces.GPU function when running on Hugging Face ZeroGPU
try:
    import spaces

    @spaces.GPU
    def gpu_predict_dummy():
        return "ZeroGPU enabled"
except ImportError:
    pass


class RestoreApiPrefix:
    """Restore /api prefix stripped by FastAPI sub-application mount."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        environ["PATH_INFO"] = "/api" + environ.get("PATH_INFO", "")
        return self.wsgi_app(environ, start_response)


flask_app = create_app()
api = FastAPI(title="SkinSenseAI API")
api.mount("/api", WSGIMiddleware(RestoreApiPrefix(flask_app)))

with gr.Blocks(title="SkinSenseAI Backend") as demo:
    gr.Markdown(
        "# SkinSenseAI Backend API\n"
        "Flask + PyTorch + OpenCV + Scikit-learn API is running.\n\n"
        "- Health check: [`/api/health`](/api/health)\n"
        "- Endpoints: `/api/analyze-image`, `/api/build-profile`, `/api/recommendations`, `/api/treatments`"
    )

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
