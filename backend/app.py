"""
app.py — Flask application entry point.
Hybrid Facial Skin Analysis & Skincare Recommendation System
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # Import torch before numpy/cv2/datasets on Windows to avoid c10.dll WinError 1114
from flask import Flask
from flask_cors import CORS

from backend.config import config
from backend.routes.analyze import analyze_bp
from backend.routes.recommend import recommend_bp
from backend.routes.profile import profile_bp
from backend.routes.treatment import treatment_bp
from backend.services.model_loader import ModelLoader


def create_app():
    import os
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

    loader = ModelLoader.get_instance()
    if not (loader.acne_model_loaded and loader.recommender_loaded and loader.treatment_loaded):
        loader.load_all()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"]      = config.UPLOAD_FOLDER

    # Allow requests from React dev server
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(analyze_bp,   url_prefix="/api")
    app.register_blueprint(recommend_bp, url_prefix="/api")
    app.register_blueprint(profile_bp,   url_prefix="/api")
    app.register_blueprint(treatment_bp, url_prefix="/api")

    # Health check
    @app.route("/api/health", methods=["GET"])
    def health():
        from flask import jsonify
        loader  = ModelLoader.get_instance()
        return jsonify({
            "status":        "ok",
            "models_loaded": {
                "acne_classifier": loader.acne_model_loaded,
                "recommender":     loader.recommender_loaded,
                "treatment_mapper":loader.treatment_loaded,
            },
            "version": "1.0.0",
        }), 200

    @app.errorhandler(413)
    def too_large(e):
        from flask import jsonify
        return jsonify({"success": False, "error": "file_too_large",
                        "message": "Image file must be under 16 MB."}), 413

    @app.errorhandler(404)
    def not_found(e):
        from flask import jsonify
        return jsonify({"success": False, "error": "not_found",
                        "message": "Endpoint not found."}), 404

    @app.errorhandler(500)
    def server_error(e):
        from flask import jsonify
        return jsonify({"success": False, "error": "server_error",
                        "message": "An internal server error occurred."}), 500

    return app


if __name__ == "__main__":
    import os
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

    # Pre-load ML models at startup
    print("[STARTUP] Loading ML models...")
    loader = ModelLoader.get_instance()
    loader.load_all()

    app = create_app()
    print(f"[STARTUP] Flask server starting on http://localhost:{config.PORT}")
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)
