"""
test_recommender.py
Stage 8 — Unit tests for RecommendationEngine & allergen negation logic.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ML.training.recommendation_engine import RecommendationEngine


def test_allergen_negation_detection():
    # "Formulated without artificial fragrances" should NOT be flagged as containing fragrance
    text_safe = "formulated without artificial fragrances and colors, parabens, sulfates, alcohol"
    assert RecommendationEngine._contains_active_allergen(text_safe, "fragrance") is False
    assert RecommendationEngine._contains_active_allergen(text_safe, "alcohol") is False

    # Actual presence should be flagged
    text_unsafe = "ingredients: water, glycerin, fragrance, limonene"
    assert RecommendationEngine._contains_active_allergen(text_unsafe, "fragrance") is True


def test_query_text_builder():
    engine = RecommendationEngine()
    profile = {
        "skin_type": "oily",
        "primary_concerns": ["acne", "pigmentation"],
        "sensitivity": "sensitive",
        "acne_severity": "Moderate",
    }
    q = engine._build_query_text(profile)
    assert "salicylic acid" in q
    assert "niacinamide" in q
    assert "centella asiatica" in q
