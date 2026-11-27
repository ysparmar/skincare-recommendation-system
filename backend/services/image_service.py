"""
image_service.py — Image processing service.
Handles face detection, skin tone estimation, and acne severity prediction.
"""

import io
import os
import sys
import pathlib
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.config import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((config.ACNE_IMAGE_SIZE, config.ACNE_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

HAAR_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

TONE_CATEGORIES = [
    ("Light",       (220, 190, 160)),
    ("Light Medium",(200, 170, 140)),
    ("Medium",      (180, 150, 120)),
    ("Tan",         (160, 120,  90)),
    ("Deep",        (120,  80,  55)),
    ("Very Deep",   ( 80,  50,  35)),
]

FITZPATRICK_TONE_MAP = {
    "Light":        "I-II",
    "Light Medium": "II-III",
    "Medium":       "III-IV",
    "Tan":          "IV-V",
    "Deep":         "V-VI",
    "Very Deep":    "VI",
}


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Load a PIL image from raw bytes."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def validate_image_quality(pil_image: Image.Image) -> dict:
    """
    Check basic image quality: resolution and blurriness.
    Returns {"valid": bool, "reason": str}.
    """
    w, h = pil_image.size
    if w < 100 or h < 100:
        return {"valid": False, "reason": "Image resolution too low (minimum 100×100 px)"}

    # Blurriness check via Laplacian variance
    cv_img  = np.array(pil_image)
    gray    = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 20:
        return {"valid": False, "reason": "Image appears too blurry for reliable analysis"}

    return {"valid": True, "reason": "ok"}


def detect_face(pil_image: Image.Image):
    """
    Detect the largest face in the image using OpenCV Haar Cascade.
    Returns (face_detected: bool, face_roi: PIL.Image or None).
    """
    cv_img = np.array(pil_image)
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    faces = HAAR_FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return False, None

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Add 15% padding around the face
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(pil_image.width,  x + w + pad_x)
    y2 = min(pil_image.height, y + h + pad_y)

    face_roi = pil_image.crop((x1, y1, x2, y2))
    return True, face_roi


def predict_acne_severity(face_roi: Image.Image, model, device) -> dict:
    """
    Run EfficientNet-B0 inference on a face crop using horizontal-flip
    Test-Time Augmentation (TTA) for improved prediction stability.
    """
    img_tensor = INFERENCE_TRANSFORM(face_roi).unsqueeze(0).to(device)
    flipped_tensor = torch.flip(img_tensor, dims=[3])

    with torch.no_grad():
        probs_orig = torch.softmax(model(img_tensor), dim=1)
        probs_flip = torch.softmax(model(flipped_tensor), dim=1)
        probs = (0.5 * (probs_orig + probs_flip)).cpu().numpy()[0]

    pred_idx    = int(np.argmax(probs))
    confidence  = float(probs[pred_idx])
    all_probs   = {name: round(float(p), 4) for name, p in zip(config.ACNE_CLASS_NAMES, probs)}

    return {
        "severity":          config.ACNE_CLASS_NAMES[pred_idx],
        "severity_level":    pred_idx,
        "confidence":        round(confidence, 4),
        "confidence_pct":    int(confidence * 100),
        "all_probabilities": all_probs,
        "low_confidence":    confidence < config.ACNE_CONFIDENCE_THRESHOLD,
    }


def estimate_skin_tone(face_roi: Image.Image, n_clusters: int = 3) -> dict:
    """
    Estimate dominant skin tone using K-Means clustering on the face region.
    Uses YCrCb color space to isolate skin pixels before clustering.

    Returns a dict with tone category, hex color, and disclaimer.
    """
    try:
        cv_img = np.array(face_roi)
        ycrcb  = cv2.cvtColor(cv_img, cv2.COLOR_RGB2YCrCb)

        # Skin pixel mask (YCrCb range is a known heuristic)
        lower = np.array([0,  133, 77],  dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)

        skin_pixels = cv_img[skin_mask > 0]
        if len(skin_pixels) < 50:
            return {"available": False, "note": "Insufficient skin pixels detected."}

        # K-Means on skin pixels
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        km.fit(skin_pixels)

        # Largest cluster is the dominant skin tone
        counts   = np.bincount(km.labels_)
        dominant = km.cluster_centers_[np.argmax(counts)].astype(int)
        r, g, b  = int(dominant[0]), int(dominant[1]), int(dominant[2])

        # Match to tone category
        tone_cat = _classify_tone(r, g, b)
        hex_color = f"#{r:02X}{g:02X}{b:02X}"

        return {
            "available":           True,
            "dominant_hex":        hex_color,
            "dominant_rgb":        [r, g, b],
            "tone_category":       tone_cat,
            "fitzpatrick_estimate":FITZPATRICK_TONE_MAP.get(tone_cat, "Unknown"),
            "note":                "Approximate estimate only. Subject to lighting and image quality conditions.",
        }

    except Exception as e:
        return {"available": False, "note": f"Skin tone estimation failed: {str(e)}"}


def _classify_tone(r: int, g: int, b: int) -> str:
    """Find the closest tone category to the given RGB value."""
    min_dist = float("inf")
    best     = "Medium"
    for label, (tr, tg, tb) in TONE_CATEGORIES:
        dist = ((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best     = label
    return best
