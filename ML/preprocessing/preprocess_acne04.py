"""
preprocess_acne04.py
Stage 4 — Preprocessing pipeline for the ACNE04 dataset.

Loads ACNE04 from HuggingFace, verifies structure, computes class
distribution and weights, and saves a preprocessed manifest CSV.

Usage:
    python ML/preprocessing/preprocess_acne04.py

Outputs:
    ML/datasets/acne04/acne04_manifest.csv   — per-image metadata
    ML/datasets/acne04/class_weights.pkl     — torch class weights
    ML/evaluation/acne04_distribution.png    — class distribution chart
"""

import os
import sys
import json
import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ACNE_DS_DIR  = ROOT / "ML" / "datasets" / "acne04"
EVAL_DIR     = ROOT / "ML" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_MAP = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Very Severe"}


def load_acne04():
    """Load ACNE04 from HuggingFace datasets library."""
    print("[INFO] Loading ACNE04 dataset from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset("Layered-Labs/ACNE04", cache_dir=str(ACNE_DS_DIR))
    print(f"[INFO] Dataset loaded. Splits available: {list(ds.keys())}")
    return ds


def verify_structure(ds):
    """Print dataset structure info and verify required columns."""
    print("\n[VERIFY] Dataset Structure:")
    for split_name, split in ds.items():
        print(f"  Split '{split_name}': {len(split)} rows")
        print(f"  Columns: {split.column_names}")
        if len(split) > 0:
            sample = split[0]
            print(f"  Sample keys: {list(sample.keys())}")
            if "level" in sample:
                print(f"  Sample level: {sample['level']}")
            if "severity" in sample:
                print(f"  Sample severity: {sample['severity']}")
    print()


def build_manifest(ds):
    """
    Build a DataFrame with per-image metadata using the canonical
    fold_0_split column for train/test assignment.
    Returns manifest DataFrame.
    """
    print("[INFO] Building image manifest...")
    records = []

    # ACNE04 on HuggingFace has a single 'train' split with fold columns
    split_data = ds["train"]

    for idx, row in enumerate(split_data):
        level    = row.get("level", -1)
        severity = row.get("severity", SEVERITY_MAP.get(level, "Unknown"))
        # Use canonical fold_0_split for train/test; fallback to 'split' column
        fold_split = row.get("fold_0_split", row.get("split", "train"))

        records.append({
            "index":          idx,
            "file_name":      row.get("file_name", f"img_{idx}"),
            "level":          level,
            "severity":       severity,
            "fold_split":     fold_split,          # 'train' or 'test'
            "imagewidth":     row.get("imagewidth", 0),
            "imageheight":    row.get("imageheight", 0),
            "num_lesions":    row.get("num_lesions_annotated", 0),
        })

    manifest = pd.DataFrame(records)
    print(f"[INFO] Manifest built: {len(manifest)} total records")
    return manifest


def analyze_distribution(manifest):
    """Print class distribution and train/test split statistics."""
    print("\n[ANALYSIS] Class Distribution (Full Dataset):")
    dist = manifest.groupby(["level", "severity"])["index"].count().reset_index()
    dist.columns = ["Level", "Severity", "Count"]
    dist["Pct"] = (dist["Count"] / len(manifest) * 100).round(1)
    print(dist.to_string(index=False))

    print("\n[ANALYSIS] Train/Test Split:")
    split_dist = manifest.groupby(["fold_split", "severity"])["index"].count().unstack(fill_value=0)
    print(split_dist)

    total = len(manifest)
    train_n = (manifest["fold_split"] == "train").sum()
    test_n  = (manifest["fold_split"] == "test").sum()
    print(f"\n  Total: {total} | Train: {train_n} ({train_n/total*100:.1f}%) | Test: {test_n} ({test_n/total*100:.1f}%)")

    return dist


def compute_class_weights(manifest):
    """
    Compute inverse-frequency class weights for WeightedRandomSampler
    and weighted cross-entropy loss.
    Returns dict: {level_int: weight_float}
    """
    train_df = manifest[manifest["fold_split"] == "train"]
    counts   = train_df["level"].value_counts().sort_index()
    n_total  = len(train_df)

    weights = {}
    print("\n[INFO] Class Weights (inverse frequency, normalized):")
    for level in sorted(counts.index):
        w = n_total / (len(counts) * counts[level])
        weights[int(level)] = round(float(w), 4)
        print(f"  Level {level} ({SEVERITY_MAP[level]}): count={counts[level]}, weight={w:.4f}")

    return weights


def save_outputs(manifest, weights, dist):
    """Save manifest CSV and class weights pickle."""
    manifest_path = ACNE_DS_DIR / "acne04_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n[SAVED] Manifest → {manifest_path}")

    weights_path = ACNE_DS_DIR / "class_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)
    print(f"[SAVED] Class weights → {weights_path}")

    # Also save as JSON for readability
    weights_json_path = ACNE_DS_DIR / "class_weights.json"
    with open(weights_json_path, "w") as f:
        json.dump({SEVERITY_MAP[k]: v for k, v in weights.items()}, f, indent=2)
    print(f"[SAVED] Class weights (JSON) → {weights_json_path}")


def plot_distribution(manifest):
    """Generate and save class distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACNE04 Dataset — Class Distribution", fontsize=14, fontweight="bold")

    severity_order = ["Mild", "Moderate", "Severe", "Very Severe"]
    colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    # Full dataset distribution
    dist_counts = manifest["severity"].value_counts().reindex(severity_order, fill_value=0)
    axes[0].bar(severity_order, dist_counts.values, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Full Dataset (1,457 images)")
    axes[0].set_xlabel("Acne Severity")
    axes[0].set_ylabel("Number of Images")
    for i, v in enumerate(dist_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Train/Test split
    train_counts = manifest[manifest["fold_split"] == "train"]["severity"].value_counts().reindex(severity_order, fill_value=0)
    test_counts  = manifest[manifest["fold_split"] == "test"]["severity"].value_counts().reindex(severity_order, fill_value=0)
    x = np.arange(len(severity_order))
    width = 0.35
    axes[1].bar(x - width/2, train_counts.values, width, label="Train", color="#2196F3", edgecolor="black", linewidth=0.8)
    axes[1].bar(x + width/2, test_counts.values, width, label="Test",  color="#FF5722", edgecolor="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(severity_order)
    axes[1].set_title("Train / Test Split (Fold 0)")
    axes[1].set_xlabel("Acne Severity")
    axes[1].set_ylabel("Number of Images")
    axes[1].legend()

    plt.tight_layout()
    out_path = EVAL_DIR / "acne04_distribution.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Distribution chart → {out_path}")


def main():
    print("=" * 60)
    print(" ACNE04 Dataset Preprocessing — Stage 4")
    print("=" * 60)

    ds       = load_acne04()
    verify_structure(ds)
    manifest = build_manifest(ds)
    dist     = analyze_distribution(manifest)
    weights  = compute_class_weights(manifest)
    save_outputs(manifest, weights, dist)
    plot_distribution(manifest)

    print("\n[DONE] ACNE04 preprocessing complete.")
    print(f"       Manifest:      {ACNE_DS_DIR / 'acne04_manifest.csv'}")
    print(f"       Class weights: {ACNE_DS_DIR / 'class_weights.pkl'}")
    print(f"       Distribution:  {EVAL_DIR / 'acne04_distribution.png'}")


if __name__ == "__main__":
    main()
