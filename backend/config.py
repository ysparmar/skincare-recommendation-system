"""
config.py — Application configuration for Flask backend.
Loads environment variables and sets defaults for local development.
"""

import os
from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR   = ROOT_DIR / "ML"

# ─── Flask ────────────────────────────────────────────────────────────────────
class Config:
    DEBUG        = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    SECRET_KEY   = os.getenv("SECRET_KEY", "skincare-dev-secret-change-in-prod")
    PORT         = int(os.getenv("PORT", 5000))

    # ─── Upload Settings ───────────────────────────────────────────────────────
    UPLOAD_FOLDER       = str(ROOT_DIR / "backend" / "uploads")
    ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "webp"}
    MAX_CONTENT_LENGTH  = 16 * 1024 * 1024   # 16 MB

    # ─── ML Model Paths ────────────────────────────────────────────────────────
    MODELS_DIR                = str(ML_DIR / "models")
    ACNE_MODEL_PATH           = str(ML_DIR / "models" / "acne_classifier_efficientnet.pt")
    TFIDF_VECTORIZER_PATH     = str(ML_DIR / "models" / "tfidf_vectorizer.pkl")
    PRODUCT_FEATURES_PATH     = str(ML_DIR / "models" / "product_features.pkl")
    PRODUCT_DATA_PATH         = str(ML_DIR / "models" / "products_clean.pkl")
    TREATMENT_DATA_PATH       = str(ML_DIR / "models" / "treatment_data.pkl")

    # ─── Dataset Paths ─────────────────────────────────────────────────────────
    PRODUCTS_CSV_PATH         = str(ML_DIR / "datasets" / "skin_care_products")
    TREATMENT_CSV_PATH        = str(ML_DIR / "datasets" / "skincare_treatment")
    ACNE_DATASET_PATH         = str(ML_DIR / "datasets" / "acne04")

    # ─── Model Settings ────────────────────────────────────────────────────────
    ACNE_IMAGE_SIZE           = 224
    ACNE_NUM_CLASSES          = 4
    ACNE_CLASS_NAMES          = ["Mild", "Moderate", "Severe", "Very Severe"]
    ACNE_CONFIDENCE_THRESHOLD = 0.40   # Below this → low-confidence fallback

    # ─── Recommendation Settings ───────────────────────────────────────────────
    MAX_RECOMMENDATIONS       = 10
    MIN_RECOMMENDATION_SCORE  = 0.10
    TOP_N_KNN                 = 20     # Candidates from KNN before hybrid reranking

    # ─── Skin Tone Settings ────────────────────────────────────────────────────
    SKIN_TONE_N_CLUSTERS      = 3     # K-Means clusters for dominant skin color

config = Config()
