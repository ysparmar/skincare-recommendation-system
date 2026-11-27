"""
profile.py — /api/build-profile route.
"""

from flask import Blueprint, request, jsonify

profile_bp = Blueprint("profile", __name__)

VALID_SKIN_TYPES  = {"dry", "oily", "normal", "combination", "sensitive"}
VALID_AGE_GROUPS  = {"teen", "adult", "senior"}
VALID_SENSITIVITY = {"sensitive", "resilient", "normal"}

VALID_CONCERNS = {
    "acne", "dryness", "pigmentation", "redness", "dark_spots",
    "open_pores", "dull_skin", "anti_aging", "uneven_texture", "sensitivity",
}


@profile_bp.route("/build-profile", methods=["POST"])
def build_profile():
    """
    POST /api/build-profile
    Validates questionnaire input and constructs a unified skin profile.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "invalid_json",
                        "message": "Request body must be valid JSON."}), 400

    errors = []

    # ── Validate skin type ────────────────────────────────────────────────────
    skin_type = str(data.get("skin_type", "")).lower().strip()
    if skin_type not in VALID_SKIN_TYPES:
        errors.append(f"skin_type must be one of: {sorted(VALID_SKIN_TYPES)}")

    # ── Validate concerns ─────────────────────────────────────────────────────
    raw_concerns = data.get("concerns", [])
    if not isinstance(raw_concerns, list):
        raw_concerns = [raw_concerns]
    concerns = [c.lower().strip() for c in raw_concerns if c]

    # ── Validate age group ────────────────────────────────────────────────────
    age_group = str(data.get("age_group", "adult")).lower().strip()
    if age_group not in VALID_AGE_GROUPS:
        age_group = "adult"

    # ── Validate sensitivity ──────────────────────────────────────────────────
    sensitivity = str(data.get("sensitivity", "normal")).lower().strip()
    if sensitivity not in VALID_SENSITIVITY:
        sensitivity = "normal"

    if errors:
        return jsonify({"success": False, "error": "validation_error",
                        "messages": errors}), 422

    # ── Extract optional image analysis ──────────────────────────────────────
    image_analysis  = data.get("image_analysis", {}) or {}
    acne_severity   = image_analysis.get("severity")
    acne_from_image = bool(acne_severity and not image_analysis.get("low_confidence", True))

    # ── Build primary concerns list ───────────────────────────────────────────
    primary_concerns = list(concerns)

    # Inject skin type as a concern if skin type maps to a concern
    type_concern_map = {
        "oily":        "oily",
        "dry":         "dryness",
        "sensitive":   "sensitivity",
    }
    mapped = type_concern_map.get(skin_type)
    if mapped and mapped not in primary_concerns:
        primary_concerns.append(mapped)

    # Add acne concern if image analysis confirms it
    if acne_from_image and "acne" not in primary_concerns:
        severity_level = image_analysis.get("severity_level", 0)
        if severity_level >= 1:  # Moderate or above
            primary_concerns.insert(0, "acne")

    # Allergies and preferences
    allergies    = [a.lower().strip() for a in data.get("allergies", []) if a]
    preferences  = [p.lower().strip() for p in data.get("product_preferences", []) if p]

    # ── Profile completeness score ────────────────────────────────────────────
    fields_filled = sum([
        bool(skin_type),
        bool(concerns),
        bool(acne_severity),
        bool(image_analysis.get("skin_tone")),
        bool(allergies),
        bool(preferences),
        bool(age_group),
        sensitivity != "normal",
    ])
    completeness = round(fields_filled / 8, 2)

    skin_profile = {
        "skin_type":           skin_type,
        "primary_concerns":    primary_concerns,
        "sensitivity":         sensitivity,
        "acne_severity":       acne_severity,
        "acne_from_image":     acne_from_image,
        "acne_confidence":     image_analysis.get("confidence"),
        "skin_tone_category":  image_analysis.get("tone_category"),
        "allergies":           allergies,
        "product_preferences": preferences,
        "age_group":           age_group,
        "profile_completeness":completeness,
    }

    return jsonify({"success": True, "skin_profile": skin_profile}), 200
