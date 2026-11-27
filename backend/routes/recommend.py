"""
recommend.py — /api/recommendations, /api/similar-products, and /api/model-info routes.
"""

import json
import pathlib
from flask import Blueprint, request, jsonify
from backend.services.model_loader import ModelLoader

recommend_bp = Blueprint("recommend", __name__)

ROOT = pathlib.Path(__file__).resolve().parents[2]


@recommend_bp.route("/recommendations", methods=["POST"])
def get_recommendations():
    """
    POST /api/recommendations
    Accepts a skin_profile dict and returns ranked product recommendations + AM/PM routine.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"success": False, "error": "invalid_json",
                        "message": "Request body must be valid JSON."}), 400

    skin_profile = data.get("skin_profile")
    if not skin_profile:
        return jsonify({"success": False, "error": "missing_profile",
                        "message": "skin_profile is required."}), 422

    if not skin_profile.get("skin_type"):
        return jsonify({"success": False, "error": "missing_skin_type",
                        "message": "skin_type is required in skin_profile."}), 422

    max_results     = min(int(data.get("max_results", 10)), 20)
    category_filter = data.get("category_filter")
    sort_by         = data.get("sort_by", "score")

    loader = ModelLoader.get_instance()
    if not loader.recommender_loaded:
        return jsonify({
            "success": False,
            "error":   "recommender_unavailable",
            "message": "Recommendation engine not yet available. "
                       "Run ML/preprocessing/preprocess_products.py first.",
        }), 503

    try:
        recommendations = loader.recommender.recommend(
            skin_profile=skin_profile,
            max_results=max_results,
            category_filter=category_filter,
            sort_by=sort_by,
        )
        routine = loader.recommender.build_routine(skin_profile)

        filters_applied = {
            "skin_type": skin_profile.get("skin_type"),
            "concerns":  skin_profile.get("primary_concerns", []),
            "category":  category_filter,
        }

        return jsonify({
            "success":         True,
            "total_found":     len(recommendations),
            "filters_applied": filters_applied,
            "recommendations": recommendations,
            "routine":         routine,
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error":   "recommendation_error",
            "message": f"Recommendation failed: {str(e)}",
        }), 500


@recommend_bp.route("/similar-products", methods=["POST"])
def similar_products():
    """
    POST /api/similar-products
    Find products similar to a given product by ingredient cosine similarity.
    """
    data = request.get_json(silent=True)
    if not data or "product_name" not in data:
        return jsonify({"success": False, "error": "missing_product_name"}), 422

    loader = ModelLoader.get_instance()
    if not loader.recommender_loaded:
        return jsonify({"success": False, "error": "recommender_unavailable"}), 503

    similar = loader.recommender.get_similar_products(
        product_name=data["product_name"],
        n=int(data.get("n", 4)),
    )

    return jsonify({"success": True, "similar_products": similar}), 200


@recommend_bp.route("/model-info", methods=["GET"])
def model_info():
    """
    GET /api/model-info
    Returns real training configuration and dataset statistics for the Model Insights view.
    """
    loader = ModelLoader.get_instance()
    cfg_path = ROOT / "ML" / "models" / "acne_training_config.json"
    acne_cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            acne_cfg = json.load(f)

    return jsonify({
        "success": True,
        "acne_model": {
            "architecture": "EfficientNet-B0 (Transfer Learning, TTA)",
            "dataset":      "ACNE04 (1,457 images: 1,165 train / 292 test)",
            "test_acc":     acne_cfg.get("test_acc"),
            "tta_acc":      acne_cfg.get("tta_acc"),
            "classes":      acne_cfg.get("class_names", []),
        },
        "recommender": {
            "method":         "TF-IDF + Cosine Similarity + KNN + Review Mentions",
            "total_products": int(len(loader.recommender.products_df)) if loader.recommender_loaded else 0,
            "total_features": int(loader.recommender.feature_matrix.shape[1]) if loader.recommender_loaded else 0,
            "reviews_used":   15040,
        },
        "treatment_mapper": {
            "dataset":       "Skincare Treatment Dataset",
            "total_records": int(len(loader.treatment_df)) if loader.treatment_df is not None else 0,
        },
    }), 200
