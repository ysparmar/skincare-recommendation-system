---
title: SkinSenseAI API
emoji: ✨
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# SkinSenseAI — Facial Skin Analysis & Skincare Recommendation System

A machine learning web application that analyzes a user's facial photo and skin questionnaire to recommend suitable Korean skincare products and general ingredient guidance.

---

## What It Does

1. **Facial Photo Analysis:** Detects the face in an uploaded photo and predicts acne severity (**Mild**, **Moderate**, **Severe**, or **Very Severe**) with a confidence score.
2. **Skin Tone Estimate (Optional):** Estimates an approximate skin tone from the photo for visual context.
3. **Skin Questionnaire:** Collects skin type (Dry, Oily, Normal, Combination, Sensitive), main concerns, allergies, product preferences, and budget.
4. **Product Recommendations:** Ranks Korean skincare products that best match the user's skin profile and explains *why* each product was recommended.
5. **Treatment Guide:** Maps skin concerns to common active ingredients and expected benefits.

---

## Architecture

```text
User (Browser)
      │
      ▼
React Frontend (Upload Photo + Questionnaire + Results UI)
      │  REST API (HTTP / JSON)
      ▼
Flask Backend (Python)
      ├── Face Detection & Skin Tone (OpenCV + K-Means)
      ├── Acne Severity Model (EfficientNet-B0 / PyTorch)
      ├── Product Recommendation Engine (TF-IDF + Cosine Similarity + KNN)
      └── Treatment Lookup (Concern → Ingredients Mapping)
```

---

## How It Works (In Simple Terms)

1. **Reading the Photo:**
   - When you upload a photo, OpenCV checks image quality and locates your face.
   - A trained **EfficientNet-B0** image model classifies acne severity into one of 4 levels.
   - If enabled, **K-Means clustering** groups skin pixels to estimate an approximate skin tone.
2. **Combining With Your Answers:**
   - Your questionnaire answers (skin type, concerns, allergies, budget) are combined with the photo analysis into one **Skin Profile**.
   - If you skip the photo or the confidence is low, the system still works using your questionnaire.
3. **Finding Matching Products:**
   - Product descriptions and types are converted into text vectors (**TF-IDF**).
   - **KNN** and **Cosine Similarity** find products whose descriptions and active ingredients match your skin concerns, while filtering out your budget limits and allergens.
4. **Showing Clear Results:**
   - The app displays your skin profile, ranked product cards with match percentages and reasons, and a treatment ingredient guide.

---

## Datasets Used

| Dataset | Purpose | Size |
|---|---|---|
| **ACNE04** (HuggingFace: `Layered-Labs/ACNE04`) | Training & testing the acne severity image model | 1,457 facial images (4 severity levels) |
| **Korean Skincare Products** (Kaggle: `taniadh/skin-care`) | Product catalog & customer reviews for recommendations | 237 products, 15,040 reviews |
| **Skincare Treatment Dataset** (Kaggle: `shijo96john/skincare-treatment-dataset`) | Mapping skin concerns to active ingredients & effects | 1,120 rows (917 unique records) |

---

## How to Run Locally

### 1. Start the Backend
Double-click `run_backend.bat` or run:
```powershell
.venv\Scripts\python.exe backend\app.py
```
Runs at `http://localhost:5000`

### 2. Start the Frontend
Double-click `run_frontend.bat` or run:
```powershell
cd frontend
npm start
```
Opens at `http://localhost:3000`

---

## Disclaimer

This project is built for educational purposes only. It does **not** provide medical advice or diagnosis and does not replace a dermatologist.
