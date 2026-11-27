# PROJECT_PROGRESS.md
# Hybrid Facial Skin Analysis & Personalized Skincare Recommendation System

> **CONTINUITY NOTICE:** Any new AI session or model must read this file completely before continuing development. Do not restart the project. Continue from the last verified checkpoint.

---

## Project Overview

**Project Title:** Hybrid Facial Skin Analysis & Personalized Skincare Recommendation System  
**Subject:** BTech Computer Science & Engineering — Machine Learning  
**Type:** Full-stack ML application (React + Flask + PyTorch)  

**Objective:**  
Build a hybrid system that combines facial image analysis, user questionnaire, and ML-based product recommendation to deliver personalized skincare advice.

---

## Dataset Sources & Verified Audit

| Dataset | URL | Status | Verified Details |
|---|---|---|---|
| Skincare Treatment Dataset | https://www.kaggle.com/datasets/shijo96john/skincare-treatment-dataset | [x] Verified & Preprocessed | `Skincare Treatment Dataset.csv`: 1,120 raw rows → 917 unique records (203 duplicates removed), 10 concerns, 4 skin types, 0 nulls. |
| Korean Skincare Products | https://www.kaggle.com/datasets/taniadh/skin-care | [x] Verified & Preprocessed | `productlist.csv` (237 products) + `productReviews.csv` (15,040 reviews aggregated across 232 products), 3,864 TF-IDF features, price `$2.50–$60.00`. |
| ACNE04 (HuggingFace) | https://huggingface.co/datasets/Layered-Labs/ACNE04 | [x] Verified & Trained | 1,457 images (Mild: 513, Moderate: 633, Severe: 182, Very Severe: 129). Canonical Fold 0: 1,165 train / 292 test. |

---

## Technology Stack (Finalized)

| Component | Technology |
|---|---|
| Frontend | React 18, JavaScript, CSS, Axios, React Router |
| Backend | Python 3.11.18 (`.venv`), Flask 3.0.3, Flask-CORS |
| Deep Learning | PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128 (EfficientNet-B0 on RTX 5060 GPU) |
| Machine Learning | Scikit-learn 1.5.1 (KNN, TF-IDF, Cosine Similarity, K-Means) |
| Image Processing | OpenCV 4.10, Pillow |
| Data Processing | Pandas 2.2.2, NumPy |
| Visualization | Matplotlib 3.9.2, Seaborn 0.13.2 |
| Model Storage | PyTorch (`.pt`), Pickle (`.pkl`) |
| API | REST (Flask) |

---

## Overall Progress

**Completion: 100%** (All 9 development stages completed and verified)

---

## Development Stages & Task Checklist

### Stage 1 — Repository Analysis
**Status: ✅ COMPLETED**
- [x] Inspect reference GitHub repository architecture
- [x] Identify key ML algorithms (EfficientNet-B0, K-Means, KNN, TF-IDF)
- [x] Document technology stack & frontend/backend gap

### Stage 2 — Dataset Analysis
**Status: ✅ COMPLETED**
- [x] Audit all 3 datasets against actual downloaded files
- [x] Verify ACNE04 structure, labels, and canonical `fold_0_split` (1,165 train / 292 test)

### Stage 3 — Architecture Planning
**Status: ✅ COMPLETED**
- [x] Finalize API endpoint specifications (`documentation/API_SPECIFICATION.md`)
- [x] Create project directory structure & `requirements.txt`

### Stage 4 — Data Preprocessing
**Status: ✅ COMPLETED**
- [x] `preprocess_products.py` → `products_clean.pkl`, `tfidf_vectorizer.pkl`, `product_features.pkl`, `products_eda.png`
- [x] `preprocess_treatment.py` → `treatment_data.pkl`, `treatment_lookup.pkl`, `treatment_eda.png`
- [x] `preprocess_acne04.py` → `acne04_manifest.csv`, `class_weights.pkl`, `acne04_distribution.png`

### Stage 5 — ML Model Development
**Status: ✅ COMPLETED**
- [x] Train EfficientNet-B0 on ACNE04 (RTX 5060 GPU, full backbone fine-tune, 30 max epochs / early stop at 19): **78.42% test accuracy** (`79.79%` with TTA), **79.17% macro F1**, **81.23% macro precision**, **78.46% macro recall**
- [x] Save model & evaluation artifacts: `acne_classifier_efficientnet.pt`, `acne_training_config.json`, `acne_training_curves.png`, `acne_confusion_matrix.png`, `acne_classification_report.txt`
- [x] Build & test `RecommendationEngine` (TF-IDF + Cosine + KNN + hybrid scoring + negation-aware allergen filter)

### Stage 6 — Backend Development
**Status: ✅ COMPLETED**
- [x] Implement `backend/app.py`, `backend/config.py`, `backend/services/model_loader.py`, `backend/services/image_service.py`
- [x] Implement routes: `analyze.py`, `profile.py`, `recommend.py`, `treatment.py`
- [x] Verify all 3 models load cleanly at startup (`acne_classifier: True`, `recommender: True`, `treatment_mapper: True`)

### Stage 7 — Frontend Development
**Status: ✅ COMPLETED & BUILD VERIFIED**
- [x] Implement React app (`LandingPage`, `AnalyzePage`, `ResultsPage`, `Navbar`, `Footer`, `services/api.js`)
- [x] `npm.cmd install` & `npm.cmd run build` compiled with 0 errors and 0 warnings

### Stage 8 — Integration & Testing
**Status: ✅ COMPLETED**
- [x] `pytest tests/ -v`: **12 passed in 7.13s**
- [x] End-to-end Flask test client verified `/api/health`, `/api/build-profile`, `/api/recommendations`, `/api/treatments` with live loaded models

### Stage 9 — Documentation
**Status: ✅ COMPLETED**
- [x] Complete `README.md` with architecture, setup, actual evaluation metrics, and limitations
- [x] `documentation/API_SPECIFICATION.md`
- [x] `PROJECT_PROGRESS.md` synchronized with verified state
- [x] Windows `.bat` runner scripts updated to use `.venv\Scripts\python.exe`

---

## Known Issues & Technical Notes

| ID | Issue | Status | Notes |
|---|---|---|---|
| B-01 | Python not on global PATH | ✅ Resolved | Installed Python 3.11.18 into `.venv` via `uv` with PyTorch `2.11.0+cu128` for RTX 5060 GPU. All `.bat` scripts point to `.venv\Scripts\python.exe`. |
| B-02 | Windows `c10.dll` WinError 1114 | ✅ Resolved | Imported `torch` before `numpy`/`matplotlib`/`cv2` in `train_acne_classifier.py` and `image_service.py`. |
| B-03 | `productlist.csv` schema differences | ✅ Resolved | Used `product_description` for TF-IDF/ingredient matching and aggregated `productReviews.csv` by `product_id`. |

---

*Last updated: 2026-11-26 | All stages complete and verified*
