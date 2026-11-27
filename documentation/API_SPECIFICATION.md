# API Specification
## Hybrid Facial Skin Analysis & Skincare Recommendation System

**Base URL (local dev):** `http://localhost:5000/api`  
**Frontend origin:** `http://localhost:3000`  
**CORS:** Enabled for all origins in development  

---

## Endpoints

### GET `/api/health`
Check API server status and model load status.

**Response 200:**
```json
{
  "status": "ok",
  "models_loaded": {
    "acne_classifier": true,
    "recommender": true,
    "treatment_mapper": true
  },
  "version": "1.0.0"
}
```

---

### POST `/api/analyze-image`
Upload a facial image for acne severity classification and optional skin-tone estimation.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File | ✅ | JPEG/PNG/WEBP, max 16 MB |
| `estimate_tone` | Boolean (string) | ❌ | `"true"` to run K-Means skin tone |

**Response 200:**
```json
{
  "success": true,
  "acne_analysis": {
    "severity": "Moderate",
    "severity_level": 1,
    "confidence": 0.82,
    "confidence_pct": 82,
    "all_probabilities": {
      "Mild": 0.11,
      "Moderate": 0.82,
      "Severe": 0.05,
      "Very Severe": 0.02
    },
    "low_confidence": false
  },
  "skin_tone": {
    "available": true,
    "dominant_hex": "#C8A882",
    "dominant_rgb": [200, 168, 130],
    "tone_category": "Medium",
    "fitzpatrick_estimate": "III-IV",
    "note": "Approximate estimate only. Subject to lighting and image quality."
  },
  "face_detected": true,
  "image_quality": "good"
}
```

**Response 400 (no face detected):**
```json
{
  "success": false,
  "error": "no_face_detected",
  "message": "No face was detected in the uploaded image. Please upload a clear frontal facial photograph."
}
```

**Response 400 (low quality):**
```json
{
  "success": false,
  "error": "image_quality_low",
  "message": "Image resolution or quality is too low for reliable analysis."
}
```

---

### POST `/api/build-profile`
Build a unified skin profile from questionnaire responses and optional image analysis.

**Request:** `application/json`
```json
{
  "skin_type": "oily",
  "concerns": ["acne", "pigmentation", "open_pores"],
  "sensitivity": "sensitive",
  "acne_concern_level": "moderate",
  "dryness": false,
  "pigmentation": true,
  "redness": false,
  "dark_spots": true,
  "allergies": ["fragrance", "alcohol"],
  "product_preferences": ["serum", "moisturizer", "sunscreen"],
  "budget": "medium",
  "age_group": "adult",
  "image_analysis": {
    "severity": "Moderate",
    "severity_level": 1,
    "confidence": 0.82,
    "low_confidence": false
  }
}
```

**budget values:** `"low"` (< \$20), `"medium"` (\$20–\$60), `"high"` (> \$60), `"any"`

**Response 200:**
```json
{
  "success": true,
  "skin_profile": {
    "skin_type": "oily",
    "primary_concerns": ["acne", "pigmentation", "dark_spots"],
    "sensitivity": "sensitive",
    "acne_severity": "Moderate",
    "acne_from_image": true,
    "skin_tone_category": null,
    "allergies": ["fragrance", "alcohol"],
    "product_preferences": ["serum", "moisturizer", "sunscreen"],
    "budget": "medium",
    "age_group": "adult",
    "profile_completeness": 0.90
  }
}
```

---

### POST `/api/recommendations`
Get personalized skincare product recommendations using the hybrid recommendation engine.

**Request:** `application/json`
```json
{
  "skin_profile": { ... },
  "max_results": 10,
  "category_filter": "serum",
  "sort_by": "score"
}
```

**Response 200:**
```json
{
  "success": true,
  "total_found": 10,
  "filters_applied": {
    "skin_type": "oily",
    "budget": "medium",
    "category": "serum"
  },
  "recommendations": [
    {
      "rank": 1,
      "product_name": "COSRX Advanced Snail 96 Mucin Power Essence",
      "brand": "COSRX",
      "category": "Essence",
      "price": "$22.00",
      "price_numeric": 22.0,
      "rating": 4.7,
      "ingredients_snippet": "Snail Secretion Filtrate (96%), Betaine, Panthenol...",
      "recommendation_score": 0.87,
      "score_pct": 87,
      "recommendation_reasons": [
        "High ingredient match for oily + acne-prone skin",
        "Contains Niacinamide — targets pigmentation concern",
        "Alcohol-free — matches sensitivity requirement",
        "Within medium budget range"
      ],
      "matched_concerns": ["acne", "pigmentation"],
      "matched_ingredients": ["Niacinamide", "Panthenol"],
      "avoided_ingredients": [],
      "category_match": true
    }
  ]
}
```

---

### POST `/api/treatments`
Get treatment recommendations based on skin concerns.

**Request:** `application/json`
```json
{
  "skin_type": "oily",
  "concerns": ["acne", "pigmentation"],
  "sensitivity": "sensitive",
  "age_group": "adult"
}
```

**Response 200:**
```json
{
  "success": true,
  "treatments": [
    {
      "concern": "acne",
      "recommended_ingredients": ["Salicylic Acid", "Niacinamide", "Benzoyl Peroxide"],
      "concentrations": ["2% Salicylic Acid", "10% Niacinamide"],
      "effects": ["Exfoliation", "Sebum regulation", "Anti-inflammatory"],
      "usage_notes": "Start with lower concentrations. Patch test first if sensitive.",
      "disclaimer": "This is informational only. Consult a dermatologist for medical advice."
    }
  ],
  "disclaimer": "This application does not provide medical diagnosis. Recommendations are informational only."
}
```

---

## Data Schemas

### Skin Profile Object
```typescript
interface SkinProfile {
  skin_type: "dry" | "oily" | "normal" | "combination" | "sensitive";
  primary_concerns: string[];
  sensitivity: "sensitive" | "resilient" | "normal";
  acne_severity: "Mild" | "Moderate" | "Severe" | "Very Severe" | null;
  acne_from_image: boolean;
  skin_tone_category: "light" | "medium" | "tan" | "deep" | null;
  allergies: string[];
  product_preferences: string[];
  budget: "low" | "medium" | "high" | "any";
  age_group: "teen" | "adult" | "senior";
  profile_completeness: number;  // 0.0 – 1.0
}
```

### Product Recommendation Object
```typescript
interface ProductRecommendation {
  rank: number;
  product_name: string;
  brand: string;
  category: string;
  price: string;
  price_numeric: number;
  rating: number;
  ingredients_snippet: string;
  recommendation_score: number;   // 0.0 – 1.0
  score_pct: number;              // 0 – 100
  recommendation_reasons: string[];
  matched_concerns: string[];
  matched_ingredients: string[];
  avoided_ingredients: string[];
  category_match: boolean;
}
```
