"""
model_loader.py — Singleton service for loading all ML models at startup.
"""

import os
import sys
import pickle
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import config


class ModelLoader:
    """
    Singleton pattern — loads and caches all ML artifacts on first access.
    Thread-safe for Flask's development server.
    """

    _instance = None

    def __init__(self):
        self.acne_model          = None
        self.acne_model_loaded   = False
        self.recommender         = None
        self.recommender_loaded  = False
        self.treatment_lookup    = None
        self.treatment_df        = None
        self.treatment_loaded    = False
        use_cuda = torch.cuda.is_available() and not os.getenv("SPACE_ID")
        self.device              = torch.device("cuda" if use_cuda else "cpu")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self):
        self._load_acne_model()
        self._load_recommender()
        self._load_treatment()

    # ─── Acne Classifier ───────────────────────────────────────────────────────

    def _load_acne_model(self):
        model_path = pathlib.Path(config.ACNE_MODEL_PATH)
        if not model_path.exists():
            print(f"[WARN] Acne model not found at {model_path}. Run training first.")
            return

        try:
            from torchvision import models as tv_models
            import torch.nn as nn

            checkpoint = torch.load(str(model_path), map_location=self.device)

            model = tv_models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, config.ACNE_NUM_CLASSES),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            model = model.to(self.device)

            self.acne_model        = model
            self.acne_model_loaded = True
            val_acc = checkpoint.get("val_acc", "N/A")
            print(f"[MODEL] Acne classifier loaded. Val acc: {val_acc:.4f}" if isinstance(val_acc, float) else f"[MODEL] Acne classifier loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load acne model: {e}")

    # ─── Recommendation Engine ─────────────────────────────────────────────────

    def _load_recommender(self):
        required_files = [
            pathlib.Path(config.PRODUCT_DATA_PATH),
            pathlib.Path(config.TFIDF_VECTORIZER_PATH),
            pathlib.Path(config.PRODUCT_FEATURES_PATH),
        ]
        if not all(f.exists() for f in required_files):
            missing = [str(f) for f in required_files if not f.exists()]
            print(f"[WARN] Recommender artifacts missing: {missing}. Run preprocessing first.")
            return

        try:
            sys.path.insert(0, str(ROOT / "ML" / "training"))
            from recommendation_engine import RecommendationEngine
            engine = RecommendationEngine()
            engine.load()
            self.recommender        = engine
            self.recommender_loaded = True
            print("[MODEL] Recommendation engine loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load recommender: {e}")

    # ─── Treatment Lookup ─────────────────────────────────────────────────────

    def _load_treatment(self):
        lookup_path = pathlib.Path(config.TREATMENT_DATA_PATH)
        # Also look for treatment_lookup.pkl
        lookup_pkl  = pathlib.Path(str(lookup_path).replace("treatment_data", "treatment_lookup"))

        if not lookup_path.exists() and not lookup_pkl.exists():
            print("[WARN] Treatment data not found. Run preprocessing first.")
            return

        try:
            if lookup_pkl.exists():
                with open(lookup_pkl, "rb") as f:
                    self.treatment_lookup = pickle.load(f)
            if lookup_path.exists():
                with open(lookup_path, "rb") as f:
                    self.treatment_df = pickle.load(f)
            self.treatment_loaded = True
            print("[MODEL] Treatment data loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load treatment data: {e}")
