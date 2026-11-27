"""
preprocess_treatment.py
Stage 4 — Preprocessing pipeline for the Skincare Treatment Dataset.

Loads the CSV, verifies columns, cleans data, and builds a lookup
structure mapping skin_type + concern → ingredients/treatment.

Usage:
    python ML/preprocessing/preprocess_treatment.py

Outputs:
    ML/models/treatment_data.pkl         — cleaned + indexed DataFrame
    ML/models/treatment_lookup.pkl       — nested dict for fast lookup
    ML/evaluation/treatment_eda.png      — EDA chart
    ML/datasets/skincare_treatment/column_audit.txt
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TREATMENT_DIR = ROOT / "ML" / "datasets" / "skincare_treatment"
MODELS_DIR    = ROOT / "ML" / "models"
EVAL_DIR      = ROOT / "ML" / "evaluation"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def discover_csv(directory: pathlib.Path):
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        print(f"[ERROR] No CSV files found in {directory}")
        print("        Please download the Skincare Treatment dataset from:")
        print("        https://www.kaggle.com/datasets/shijo96john/skincare-treatment-dataset")
        print(f"        and place the CSV file(s) in: {directory}")
        sys.exit(1)
    print(f"[INFO] Found CSV: {csvs[0].name}")
    return csvs[0]


def load_and_audit(csv_path: pathlib.Path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    print(f"\n[AUDIT] File: {csv_path.name}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns ({len(df.columns)}):")

    audit_lines = [f"File: {csv_path.name}", f"Rows: {len(df)}", f"Columns:"]
    for col in df.columns:
        n_null   = df[col].isna().sum()
        n_unique = df[col].nunique()
        dtype    = str(df[col].dtype)
        line = f"  {col!r:35s} dtype={dtype:12s}  nulls={n_null:4d}  unique={n_unique:5d}"
        print(line)
        audit_lines.append(line)
        if df[col].dtype == object and n_unique < 20:
            vals = str(df[col].dropna().unique().tolist())
            print(f"    Values: {vals[:200]}")
            audit_lines.append(f"    Values: {vals[:200]}")

    audit_path = TREATMENT_DIR / "column_audit.txt"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
    print(f"\n[SAVED] Column audit → {audit_path}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to our standard schema."""
    aliases = {
        "skin_type":        ["skin_type", "skintype", "skin type"],
        "skin_subtype":     ["skin_subtype", "subtype", "skin subtype"],
        "concern":          ["concern", "skin_concern", "condition", "issue"],
        "sensitivity":      ["sensitivity", "sensitive"],
        "age_group":        ["age_group", "age group", "age"],
        "ingredients":      ["ingredients", "ingredient", "active ingredients", "actives"],
        "concentrations":   ["concentrations", "concentration", "strength"],
        "effects":          ["effects", "effect", "benefits", "outcome"],
        "treatment_profile":["treatment_profile", "treatment", "profile"],
        "internal_type":    ["internal_type", "internal type"],
    }
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for std, alias_list in aliases.items():
        for alias in alias_list:
            if alias.lower() in lower_map:
                orig = lower_map[alias.lower()]
                if orig != std:
                    rename[orig] = std
                break
    if rename:
        print(f"[INFO] Renaming columns: {rename}")
        df = df.rename(columns=rename)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[INFO] Cleaning treatment dataset. Shape before: {df.shape}")
    df = df.dropna(how="all")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan).replace("None", np.nan).replace("", np.nan)

    # Standardize concern values to lowercase
    if "concern" in df.columns:
        df["concern"] = df["concern"].str.lower().str.strip()

    # Standardize skin_type
    if "skin_type" in df.columns:
        df["skin_type"] = df["skin_type"].str.lower().str.strip()

    # Drop duplicates on key columns
    key_cols = [c for c in ["skin_type", "concern", "ingredients"] if c in df.columns]
    if key_cols:
        before = len(df)
        df = df.drop_duplicates(subset=key_cols)
        print(f"  Dropped {before - len(df)} duplicates on {key_cols}")

    df = df.reset_index(drop=True)
    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


def build_lookup(df: pd.DataFrame) -> dict:
    """
    Build a nested lookup dict:
      { skin_type: { concern: [treatment_records] } }
    
    Each treatment_record is a dict with keys:
      ingredients, concentrations, effects, treatment_profile, sensitivity
    """
    lookup = {}

    for _, row in df.iterrows():
        skin_type = str(row.get("skin_type", "any")).lower().strip()
        concern   = str(row.get("concern", "general")).lower().strip()

        record = {}
        for field in ["ingredients", "concentrations", "effects", "treatment_profile", "sensitivity", "age_group"]:
            val = row.get(field)
            if pd.notna(val):
                record[field] = str(val).strip()

        # Also store in an "any" skin_type group for concern-only lookups
        for st in [skin_type, "any"]:
            lookup.setdefault(st, {})
            lookup[st].setdefault(concern, [])
            if record not in lookup[st][concern]:
                lookup[st][concern].append(record)

    return lookup


def analyze_distribution(df: pd.DataFrame):
    print("\n[ANALYSIS] Treatment Dataset Distribution:")
    if "concern" in df.columns:
        print("\n  Concern counts:")
        print(df["concern"].value_counts().to_string())
    if "skin_type" in df.columns:
        print("\n  Skin type counts:")
        print(df["skin_type"].value_counts().to_string())


def plot_eda(df: pd.DataFrame):
    cols_to_plot = [c for c in ["concern", "skin_type", "sensitivity"] if c in df.columns]
    if not cols_to_plot:
        return

    n = len(cols_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Skincare Treatment Dataset — EDA", fontsize=13, fontweight="bold")

    colors = ["#4FC3F7", "#81C784", "#FFD54F"]
    for i, col in enumerate(cols_to_plot):
        counts = df[col].value_counts()
        axes[i].barh(counts.index[::-1], counts.values[::-1], color=colors[i % len(colors)])
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Count")

    plt.tight_layout()
    out_path = EVAL_DIR / "treatment_eda.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] EDA chart → {out_path}")


def save_artifacts(df, lookup):
    with open(MODELS_DIR / "treatment_data.pkl", "wb") as f:
        pickle.dump(df, f)
    print(f"[SAVED] Treatment DataFrame → {MODELS_DIR / 'treatment_data.pkl'}")

    with open(MODELS_DIR / "treatment_lookup.pkl", "wb") as f:
        pickle.dump(lookup, f)
    print(f"[SAVED] Treatment lookup → {MODELS_DIR / 'treatment_lookup.pkl'}")

    # JSON export for readability
    concerns = df["concern"].unique().tolist() if "concern" in df.columns else []
    skin_types = df["skin_type"].unique().tolist() if "skin_type" in df.columns else []
    summary = {
        "total_records": int(len(df)),
        "concerns": concerns,
        "skin_types": skin_types,
        "columns": list(df.columns),
    }
    with open(TREATMENT_DIR / "dataset_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Dataset stats → {TREATMENT_DIR / 'dataset_stats.json'}")


def main():
    print("=" * 60)
    print(" Skincare Treatment Dataset Preprocessing — Stage 4")
    print("=" * 60)

    csv_path = discover_csv(TREATMENT_DIR)
    df       = load_and_audit(csv_path)
    df       = normalize_columns(df)
    df       = clean_dataframe(df)
    analyze_distribution(df)
    lookup   = build_lookup(df)
    plot_eda(df)
    save_artifacts(df, lookup)

    print("\n[DONE] Treatment dataset preprocessing complete.")
    print(f"       Records:   {len(df)}")
    print(f"       Concerns:  {df['concern'].nunique() if 'concern' in df.columns else 'N/A'}")


if __name__ == "__main__":
    main()
