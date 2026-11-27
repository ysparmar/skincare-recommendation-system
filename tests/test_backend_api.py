"""
test_backend_api.py
Stage 8 — Automated API integration tests for Flask backend.
"""

import io
import json
import pathlib
import sys
import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert "version" in data


def test_not_found_handler(client):
    res = client.get("/api/nonexistent-endpoint")
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False


def test_build_profile_valid(client):
    payload = {
        "skin_type": "oily",
        "concerns": ["acne", "pigmentation"],
        "sensitivity": "sensitive",
        "allergies": ["fragrance"],
        "product_preferences": ["serum", "moisturizer"],
        "age_group": "adult",
    }
    res = client.post("/api/build-profile", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    profile = data["skin_profile"]
    assert profile["skin_type"] == "oily"
    assert "acne" in profile["primary_concerns"]
    assert "budget" not in profile
    assert 0.0 <= profile["profile_completeness"] <= 1.0


def test_model_info_endpoint(client):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "acne_model" in data
    assert "recommender" in data


def test_build_profile_invalid_skin_type(client):
    res = client.post("/api/build-profile", json={"skin_type": "invalid_type"})
    assert res.status_code == 422
    data = res.get_json()
    assert data["success"] is False
    assert data["error"] == "validation_error"


def test_build_profile_missing_json(client):
    res = client.post("/api/build-profile", data="not json")
    assert res.status_code == 400


def test_analyze_image_no_file(client):
    res = client.post("/api/analyze-image")
    assert res.status_code == 400
    data = res.get_json()
    assert data["error"] == "no_file"


def test_analyze_image_invalid_extension(client):
    data = {"image": (io.BytesIO(b"fake"), "test.txt")}
    res = client.post("/api/analyze-image", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.get_json()["error"] == "invalid_file"


def test_analyze_image_too_small(client):
    img = Image.new("RGB", (50, 50), color=(200, 180, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    data = {"image": (buf, "small.jpg")}
    res = client.post("/api/analyze-image", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.get_json()["error"] == "image_quality_low"


def test_analyze_image_no_face(client):
    # Create a textured 300x300 image with no face
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    data = {"image": (buf, "noise.jpg")}
    res = client.post("/api/analyze-image", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.get_json()["error"] == "no_face_detected"


def test_recommendations_missing_profile(client):
    res = client.post("/api/recommendations", json={})
    assert res.status_code == 422
    assert res.get_json()["error"] == "missing_profile"
