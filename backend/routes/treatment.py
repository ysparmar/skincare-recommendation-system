"""
treatment.py — /api/treatments route.
"""

import re
from flask import Blueprint, request, jsonify
from backend.services.model_loader import ModelLoader

treatment_bp = Blueprint("treatment", __name__)

DISCLAIMER = (
    "This information is for educational purposes only. "
    "It does not constitute medical advice or diagnosis. "
    "Consult a licensed dermatologist for clinical evaluation and treatment."
)


def _split_tokens(raw: str, split_plus: bool = True) -> list:
    if not raw:
        return []
    pattern = r"\s*[+,]\s*" if split_plus else r"\s*,\s*"
    return [t.strip() for t in re.split(pattern, raw) if t.strip()]


@treatment_bp.route("/treatments", methods=["POST"])
def get_treatments():
    """
    POST /api/treatments
    Maps skin concerns to treatment recommendations from the Skincare Treatment dataset.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json"}), 400

    skin_type   = str(data.get("skin_type",  "normal")).lower().strip()
    concerns    = [c.lower().strip() for c in data.get("concerns", []) if c]
    sensitivity = str(data.get("sensitivity", "normal")).lower().strip()

    loader = ModelLoader.get_instance()

    if not loader.treatment_loaded:
        return jsonify({
            "success":    False,
            "error":      "treatment_data_unavailable",
            "message":    "Treatment data not loaded. Run preprocessing first.",
            "disclaimer": DISCLAIMER,
        }), 503

    lookup = loader.treatment_lookup or {}
    treatments = []

    for concern in (concerns or ["general"]):
        concern_clean = concern.replace("_", " ")
        records = (
            lookup.get(skin_type, {}).get(concern_clean, []) or
            lookup.get("any",     {}).get(concern_clean, []) or
            []
        )

        if not records:
            for key in lookup.get(skin_type, {}):
                if concern_clean in key or key in concern_clean:
                    records = lookup[skin_type][key]
                    break

        if not records:
            for key in lookup.get("any", {}):
                if concern_clean in key or key in concern_clean:
                    records = lookup["any"][key]
                    break

        if not records:
            continue

        # Prefer sensitivity-matching record if available
        target_sens = "yes" if sensitivity == "sensitive" else "no"
        matching_sens = [r for r in records if r.get("sensitivity", "").lower() == target_sens]
        rec = matching_sens[0] if matching_sens else records[0]

        ing_list  = _split_tokens(rec.get("ingredients", ""), split_plus=True)
        conc_list = _split_tokens(rec.get("concentrations", ""), split_plus=True)
        eff_list  = _split_tokens(rec.get("effects", ""), split_plus=False)

        usage_notes = []
        if sensitivity == "sensitive":
            usage_notes.append("Start with lower concentrations. Perform a patch test first.")
        usage_notes.append("Introduce new actives gradually into your skincare routine.")

        treatments.append({
            "concern":                 concern,
            "recommended_ingredients": ing_list[:8],
            "concentrations":          conc_list[:6],
            "effects":                 eff_list[:6],
            "treatment_profile":       rec.get("treatment_profile", ""),
            "usage_notes":             usage_notes,
            "disclaimer":              DISCLAIMER,
        })

    if not treatments:
        treatments = [{
            "concern":    concern if concerns else "general",
            "note":       "No specific treatment data found for this concern in the dataset.",
            "disclaimer": DISCLAIMER,
        } for concern in (concerns[:3] if concerns else ["general"])]

    return jsonify({
        "success":    True,
        "treatments": treatments,
        "disclaimer": DISCLAIMER,
    }), 200
