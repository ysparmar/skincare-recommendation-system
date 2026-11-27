"""
recommendation_engine.py
Stage 5 — Hybrid Skincare Product Recommendation Engine.

Uses TF-IDF + Cosine Similarity + KNN to find candidate products,
then applies a hybrid scoring function that integrates:
  - Ingredient & description similarity to skin concerns
  - Skin type compatibility (concern-to-ingredient matching)
  - Customer review skin-type mention boost (from 15,040 reviews)
  - Category affinity & negation-aware allergen filtering
"""

import sys
import json
import pickle
import pathlib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "ML" / "models"


CONCERN_INGREDIENT_MAP = {
    "acne": [
        "salicylic acid", "benzoyl peroxide", "niacinamide", "zinc",
        "tea tree", "sulfur", "azelaic acid", "retinol", "retinal",
        "glycolic acid", "lactic acid", "centella asiatica", "cica", "breakouts",
    ],
    "pigmentation": [
        "niacinamide", "vitamin c", "ascorbic acid", "kojic acid",
        "arbutin", "licorice", "tranexamic acid", "azelaic acid",
        "glycolic acid", "lactic acid", "retinol", "retinal", "brightening",
    ],
    "dryness": [
        "hyaluronic acid", "sodium hyaluronate", "glycerin", "ceramide",
        "squalane", "shea butter", "panthenol", "aloe", "hydrating",
    ],
    "redness": [
        "centella asiatica", "cica", "aloe", "allantoin", "azulene",
        "niacinamide", "green tea", "chamomile", "azelaic acid", "calming",
    ],
    "dark_spots": [
        "vitamin c", "ascorbic acid", "niacinamide", "kojic acid",
        "arbutin", "tranexamic acid", "retinol", "glycolic acid", "dark spots",
    ],
    "open_pores": [
        "niacinamide", "salicylic acid", "retinol", "retinal", "glycolic acid",
        "zinc", "clay", "kaolin", "pores",
    ],
    "dull_skin": [
        "vitamin c", "ascorbic acid", "glycolic acid", "lactic acid",
        "niacinamide", "retinol", "aha", "bha", "pha", "brighter",
    ],
    "oily": [
        "niacinamide", "salicylic acid", "zinc", "clay", "kaolin",
        "witch hazel", "retinol", "glycolic acid", "sebum",
    ],
    "sensitive": [
        "centella asiatica", "cica", "aloe", "allantoin", "panthenol",
        "ceramide", "hyaluronic acid", "oat", "chamomile", "soothing",
    ],
    "anti_aging": [
        "retinol", "retinal", "bakuchiol", "vitamin c", "peptide", "niacinamide",
        "hyaluronic acid", "ceramide", "collagen", "wrinkles",
    ],
    "uneven_texture": [
        "glycolic acid", "lactic acid", "salicylic acid", "aha", "bha",
        "pha", "retinol", "retinal", "niacinamide", "texture",
    ],
}

ALLERGEN_NEGATION_PREFIXES = ("without ", "free of ", "no ", "formulated without ")

SENSITIZING_INGREDIENTS = [
    "fragrance", "parfum", "alcohol denat", "denatured alcohol",
    "sodium lauryl sulfate", "sls", "essential oil",
]

SKIN_TYPE_CATEGORIES = {
    "oily":        ["toner", "serum", "gel", "essence", "sun protection", "cleanser"],
    "dry":         ["moisturizer", "cream", "essence", "serum", "mask", "cleanser"],
    "combination": ["toner", "serum", "moisturizer", "essence", "cleanser"],
    "normal":      ["serum", "moisturizer", "essence", "toner", "sun protection"],
    "sensitive":   ["moisturizer", "serum", "essence", "sun protection", "mask"],
}

AM_ROUTINE_STEPS = [
    ("Cleanser",       ["cleanser"]),
    ("Toner",          ["toner"]),
    ("Serum / Essence",["serum", "essence"]),
    ("Moisturizer",    ["moisturizer", "eye treatment"]),
    ("Sun Protection", ["sun protection"]),
]

PM_ROUTINE_STEPS = [
    ("Cleanser",            ["cleanser"]),
    ("Toner",               ["toner"]),
    ("Targeted Treatment",  ["other/spot treatments", "exfoliator", "serum", "essence"]),
    ("Moisturizer / Mask",  ["moisturizer", "mask", "sheet mask", "lip treatment"]),
]


class RecommendationEngine:
    """
    Hybrid skincare product recommendation engine.
    Loads pre-built TF-IDF features and applies multi-factor scoring.
    """

    def __init__(self):
        self.products_df    = None
        self.vectorizer     = None
        self.feature_matrix = None
        self.knn_model      = None
        self._loaded        = False

    def load(self):
        """Load all preprocessing artifacts from disk."""
        required = {
            "products_clean.pkl":   "products_df",
            "tfidf_vectorizer.pkl": "vectorizer",
            "product_features.pkl": "feature_matrix",
        }

        for fname, attr in required.items():
            fpath = MODELS_DIR / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Missing model artifact: {fpath}\n"
                    f"Run: python ML/preprocessing/preprocess_products.py"
                )
            with open(fpath, "rb") as f:
                setattr(self, attr, pickle.load(f))

        n_neighbors = min(20, len(self.products_df))
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1
        )
        self.knn_model.fit(self.feature_matrix)

        self._loaded = True
        print(f"[ENGINE] Loaded {len(self.products_df)} products.")
        print(f"[ENGINE] Feature matrix: {self.feature_matrix.shape}")

    def recommend(self, skin_profile: dict, max_results: int = 10,
                  category_filter: str = None, sort_by: str = "score") -> list:
        if not self._loaded:
            self.load()

        query_text = self._build_query_text(skin_profile)
        query_vec  = self.vectorizer.transform([query_text])

        n_candidates = min(80, len(self.products_df))
        distances, indices = self.knn_model.kneighbors(query_vec, n_neighbors=n_candidates)
        candidate_indices  = indices[0]
        candidate_sims     = 1 - distances[0]

        candidates = self._apply_filters(
            candidate_indices, candidate_sims, skin_profile, category_filter
        )

        scored = self._score_candidates(candidates, skin_profile)
        scored.sort(key=lambda x: x["recommendation_score"], reverse=True)

        results = scored[:max_results]
        for rank, item in enumerate(results, 1):
            item["rank"] = rank

        return results

    def build_routine(self, skin_profile: dict) -> dict:
        """Build structured AM and PM routines from top-ranked products."""
        top_products = self.recommend(skin_profile, max_results=40)

        def pick_steps(step_defs):
            picked = []
            used_names = set()
            for step_name, cat_keywords in step_defs:
                match = None
                for prod in top_products:
                    cat = prod["category"].lower()
                    if any(k in cat for k in cat_keywords) and prod["product_name"] not in used_names:
                        match = prod
                        break
                if match:
                    used_names.add(match["product_name"])
                    picked.append({
                        "step":         step_name,
                        "product_name": match["product_name"],
                        "brand":        match["brand"],
                        "category":     match["category"],
                        "score_pct":    match["score_pct"],
                        "reason":       match["recommendation_reasons"][0] if match["recommendation_reasons"] else "",
                    })
            return picked

        return {
            "am": pick_steps(AM_ROUTINE_STEPS),
            "pm": pick_steps(PM_ROUTINE_STEPS),
        }

    def _build_query_text(self, profile: dict) -> str:
        tokens = []

        for concern in profile.get("primary_concerns", []):
            concern_key = concern.lower().replace(" ", "_")
            tokens.extend(CONCERN_INGREDIENT_MAP.get(concern_key, []))

        skin_type = profile.get("skin_type", "normal")
        tokens.extend(CONCERN_INGREDIENT_MAP.get(skin_type.lower(), []))

        severity = (profile.get("acne_severity") or "").lower()
        if severity in ["moderate", "severe", "very severe"]:
            tokens.extend(CONCERN_INGREDIENT_MAP.get("acne", []))

        if profile.get("sensitivity") == "sensitive":
            tokens.extend(CONCERN_INGREDIENT_MAP.get("sensitive", []))

        return " ".join(tokens) if tokens else "moisturizer hyaluronic acid"

    @staticmethod
    def _contains_active_allergen(ing_text: str, allergen: str) -> bool:
        """
        Return True only if allergen appears without a negation context
        like 'formulated without fragrance' or 'free of alcohol' in the same sentence.
        """
        if allergen not in ing_text:
            return False
        idx = ing_text.find(allergen)
        window = ing_text[max(0, idx - 200):idx]
        if "." in window:
            window = window.rsplit(".", 1)[-1]
        if any(prefix in window for prefix in ALLERGEN_NEGATION_PREFIXES):
            return False
        return True

    def _apply_filters(self, indices, similarities, profile, category_filter):
        allergies = [a.lower() for a in profile.get("allergies", [])]

        candidates = []
        for idx, sim in zip(indices, similarities):
            row = self.products_df.iloc[idx]

            ing_text = str(row.get("ingredients", "")).lower()
            if any(self._contains_active_allergen(ing_text, a) for a in allergies):
                continue

            if category_filter:
                cat = str(row.get("category", "")).lower()
                if category_filter.lower() not in cat:
                    continue

            candidates.append({"idx": idx, "cosine_sim": float(sim), "row": row})

        return candidates

    def _score_candidates(self, candidates, profile) -> list:
        concerns       = [c.lower().replace(" ", "_") for c in profile.get("primary_concerns", [])]
        skin_type      = profile.get("skin_type", "normal").lower()
        sensitivity    = profile.get("sensitivity", "normal").lower()
        preferred_cats = SKIN_TYPE_CATEGORIES.get(skin_type, [])
        user_prefs     = [p.lower() for p in profile.get("product_preferences", [])]

        results = []
        for cand in candidates:
            row      = cand["row"]
            ing_text = str(row.get("ingredients", "")).lower()
            category = str(row.get("category", "")).lower()
            rev_cnt  = int(row.get("review_count", 0)) if "review_count" in row and pd.notna(row.get("review_count")) else 0
            mentions = row.get("review_skin_mentions") if isinstance(row.get("review_skin_mentions"), dict) else {}

            reasons             = []
            matched_concerns    = []
            matched_ingredients = []
            avoided_ingredients = []

            cosine_score = cand["cosine_sim"]

            concern_score = 0.0
            for concern in concerns:
                beneficial = CONCERN_INGREDIENT_MAP.get(concern, [])
                for ing in beneficial:
                    if ing in ing_text:
                        concern_score += 1.0
                        matched_ingredients.append(ing.title())
                        if concern not in matched_concerns:
                            matched_concerns.append(concern)
                        reasons.append(f"Mentions {ing.title()} — targets {concern.replace('_', ' ')}")
            concern_score = min(concern_score / max(len(concerns) * 3, 1), 1.0)

            cat_score = 0.0
            for preferred in (user_prefs or preferred_cats):
                if preferred in category:
                    cat_score = 1.0
                    reasons.append(f"Category '{row.get('category', category)}' matches your preferences")
                    break
            if cat_score == 0.0:
                for preferred in preferred_cats:
                    if preferred in category:
                        cat_score = 0.7
                        reasons.append(f"Category '{row.get('category', category)}' suits {skin_type} skin")
                        break

            # Review skin-type mention boost (from 15,040 customer reviews)
            skin_mention_count = int(mentions.get(skin_type, 0))
            if skin_mention_count > 0:
                reasons.append(f"Mentioned in {skin_mention_count} {skin_type}-skin customer reviews")
            review_boost = min(skin_mention_count / 25.0, 1.0)

            sensitizing_penalty = 0.0
            if sensitivity == "sensitive":
                for s_ing in SENSITIZING_INGREDIENTS:
                    if self._contains_active_allergen(ing_text, s_ing):
                        sensitizing_penalty += 0.15
                        avoided_ingredients.append(s_ing.title())
            sensitizing_penalty = min(sensitizing_penalty, 0.30)

            final_score = (
                0.40 * cosine_score +
                0.30 * concern_score +
                0.15 * cat_score +
                0.15 * review_boost -
                sensitizing_penalty
            )
            final_score = max(0.0, min(1.0, final_score))

            if not reasons:
                reasons.append("Product description aligns with your skin profile")

            review_snippet = ""
            if "review_text" in row and pd.notna(row.get("review_text")):
                review_snippet = str(row.get("review_text"))[:200]

            results.append({
                "product_name":           str(row.get("product_name", "Unknown Product")),
                "brand":                  str(row.get("brand", "Unknown Brand")),
                "category":               str(row.get("category", "Skincare")),
                "review_count":           rev_cnt,
                "skin_type_mentions":     skin_mention_count,
                "ingredients_snippet":    str(row.get("ingredients", ""))[:300],
                "recommendation_score":   round(final_score, 4),
                "score_pct":              int(final_score * 100),
                "recommendation_reasons": list(dict.fromkeys(reasons))[:5],
                "matched_concerns":       list(dict.fromkeys(matched_concerns)),
                "matched_ingredients":    list(dict.fromkeys(matched_ingredients))[:8],
                "avoided_ingredients":    list(dict.fromkeys(avoided_ingredients)),
                "category_match":         bool(cat_score > 0),
                "review_snippet":         review_snippet,
            })

        return results

    def get_similar_products(self, product_name: str, n: int = 4) -> list:
        if not self._loaded:
            self.load()

        col = "product_name" if "product_name" in self.products_df.columns else self.products_df.columns[0]
        matches = self.products_df[self.products_df[col].str.lower() == product_name.lower()]
        if matches.empty:
            return []

        prod_idx = matches.index[0]
        prod_vec = self.feature_matrix[prod_idx]
        sims     = cosine_similarity(prod_vec, self.feature_matrix)[0]
        top_idx  = np.argsort(sims)[::-1][1:n+1]

        return [
            {
                "product_name": str(self.products_df.iloc[i].get("product_name", "")),
                "brand":        str(self.products_df.iloc[i].get("brand", "")),
                "category":     str(self.products_df.iloc[i].get("category", "")),
                "similarity":   round(float(sims[i]), 4),
            }
            for i in top_idx
        ]


if __name__ == "__main__":
    engine = RecommendationEngine()
    engine.load()

    test_profile = {
        "skin_type":        "oily",
        "primary_concerns": ["acne", "pigmentation"],
        "sensitivity":      "sensitive",
        "acne_severity":    "Moderate",
        "allergies":        ["fragrance"],
        "age_group":        "adult",
    }

    results = engine.recommend(test_profile, max_results=5)
    for r in results:
        print(f"Rank {r['rank']}: {r['product_name']} ({r['brand']}) | {r['score_pct']}% | {r['skin_type_mentions']} oily mentions")
