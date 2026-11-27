"""
analyze.py — /api/analyze-image route.
"""

import os
import uuid
from flask import Blueprint, request, jsonify
from backend.config import config
from backend.services.model_loader import ModelLoader
from backend.services.image_service import (
    load_image_from_bytes,
    validate_image_quality,
    detect_face,
    predict_acne_severity,
    estimate_skin_tone,
)

analyze_bp = Blueprint("analyze", __name__)

ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@analyze_bp.route("/analyze-image", methods=["POST"])
def analyze_image():
    """
    POST /api/analyze-image
    Accepts a multipart image upload.
    Runs face detection, acne severity classification, and optional skin tone estimation.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "no_file",
                        "message": "No image file provided in the request."}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"success": False, "error": "invalid_file",
                        "message": "Please upload a valid image file (JPG, PNG, or WEBP)."}), 400

    estimate_tone = request.form.get("estimate_tone", "false").lower() == "true"

    try:
        file_bytes = file.read()
        pil_image  = load_image_from_bytes(file_bytes)

        # ── Image quality check ───────────────────────────────────────────────
        quality = validate_image_quality(pil_image)
        if not quality["valid"]:
            return jsonify({
                "success": False,
                "error":   "image_quality_low",
                "message": quality["reason"],
            }), 400

        # ── Face detection ────────────────────────────────────────────────────
        face_detected, face_roi = detect_face(pil_image)

        if not face_detected:
            return jsonify({
                "success":       False,
                "error":         "no_face_detected",
                "message":       "No face was detected in the uploaded image. "
                                 "Please upload a clear frontal facial photograph.",
                "face_detected": False,
            }), 400

        # ── Acne severity prediction ──────────────────────────────────────────
        loader = ModelLoader.get_instance()

        acne_result = None
        if loader.acne_model_loaded:
            acne_result = predict_acne_severity(face_roi, loader.acne_model, loader.device)
        else:
            # Graceful fallback when model not yet trained
            acne_result = {
                "severity":          "Unknown",
                "severity_level":    -1,
                "confidence":        0.0,
                "confidence_pct":    0,
                "all_probabilities": {},
                "low_confidence":    True,
                "note":              "Acne classifier not yet available. "
                                     "Run training pipeline first.",
            }

        # ── Skin tone estimation (optional) ───────────────────────────────────
        tone_result = None
        if estimate_tone:
            tone_result = estimate_skin_tone(face_roi)

        # ── Save uploaded image with unique filename ──────────────────────────
        ext          = file.filename.rsplit(".", 1)[1].lower()
        unique_name  = f"{uuid.uuid4().hex}.{ext}"
        save_path    = os.path.join(config.UPLOAD_FOLDER, unique_name)
        with open(save_path, "wb") as f_out:
            f_out.write(file_bytes)

        return jsonify({
            "success":        True,
            "acne_analysis":  acne_result,
            "skin_tone":      tone_result,
            "face_detected":  True,
            "image_quality":  "good",
            "image_filename": unique_name,
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error":   "processing_error",
            "message": f"Image processing failed: {str(e)}",
        }), 500
