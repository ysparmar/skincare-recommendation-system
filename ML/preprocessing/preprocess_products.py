"""
preprocess_products.py
Stage 4 — Preprocessing pipeline for the Korean Skincare Products dataset.

Loads productlist.csv and productReviews.csv, inspects actual columns,
aggregates reviews per product_id, cleans the data, engineers features,
and builds TF-IDF vectors on product descriptions/ingredients.

Usage:
    python ML/preprocessing/preprocess_products.py

Outputs:
    ML/models/products_clean.pkl        — cleaned DataFrame
    ML/models/tfidf_vectorizer.pkl      — fitted TF-IDF vectorizer
    ML/models/product_features.pkl      — TF-IDF feature matrix (sparse)
    ML/evaluation/products_eda.png      — EDA chart
    ML/datasets/skin_care_products/column_audit.txt — actual column names
"""

import sys
import json
import pickle
import pathlib
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PRODUCTS_DIR = ROOT / "ML" / "datasets" / "skin_care_products"
MODELS_DIR   = ROOT / "ML" / "models"
EVAL_DIR     = ROOT / "ML" / "evaluation"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_ALIASES = {
    "product_id":          ["product_id", "product_ID", "id"],
    "product_name":        ["product_name", "name", "product name", "productname", "title"],
    "brand":               ["product_brand", "brand", "brand_name", "manufacturer"],
    "category":            ["product_type", "category", "type", "product type"],
    "price":               ["price", "price_usd", "price (usd)", "cost"],
    "product_description": ["product_description", "description"],
    "ingredients":         ["ingredients", "ingredient_list", "ingredient list", "inci"],
    "rating":              ["rating", "avg_rating", "average_rating", "stars", "score"],
    "review_text":         ["review", "review_text", "reviews", "comment"],
}


def discover_csvs(directory: pathlib.Path):
    csvs = sorted(directory.glob("*.csv"))
    if not csvs:
        print(f"[ERROR] No CSV files found in {directory}")
        sys.exit(1)
    print(f"[INFO] Found {len(csvs)} CSV file(s):")
    for c in csvs:
        print(f"       {c.name}")
    return csvs


def load_and_audit(csvs):
    dfs = []
    audit_lines = []

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

        # Drop unnamed index column if present
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)

        audit_lines.append(f"\n=== File: {csv_path.name} ===")
        audit_lines.append(f"Rows: {len(df)}")
        audit_lines.append(f"Columns ({len(df.columns)}):")
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_null = int(df[col].isna().sum())
            n_unique = int(df[col].nunique())
            audit_lines.append(f"  {col!r:40s} dtype={dtype:12s}  nulls={n_null:5d}  unique={n_unique:6d}")

        dfs.append((csv_path.name, df))

    audit_path = PRODUCTS_DIR / "column_audit.txt"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
    print(f"\n[SAVED] Column audit -> {audit_path}")
    print("\n".join(audit_lines))

    return dfs


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map actual column names to our standardized names using COLUMN_ALIASES."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}

    for std_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_map:
                orig = lower_map[alias.lower()]
                if orig != std_name and std_name not in rename.values():
                    rename[orig] = std_name
                break

    if rename:
        print(f"[INFO] Column renaming: {rename}")
        df = df.rename(columns=rename)

    return df


def merge_dataframes(dfs):
    """
    Merge productlist.csv and productReviews.csv on product_id.
    Aggregates reviews per product_id so each row remains one unique product.
    """
    normalized = [(name, normalize_columns(df)) for name, df in dfs]

    if len(normalized) == 1:
        return normalized[0][1]

    # Identify product list df vs reviews df
    products_df = None
    reviews_df = None
    for name, df in normalized:
        if "product_name" in df.columns:
            products_df = df
        elif "review_text" in df.columns:
            reviews_df = df

    if products_df is None:
        products_df = normalized[0][1]

    if reviews_df is not None and "product_id" in products_df.columns and "product_id" in reviews_df.columns:
        products_df["product_id"] = products_df["product_id"].astype(str).str.strip()
        reviews_df["product_id"] = reviews_df["product_id"].astype(str).str.strip()

        reviews_clean = reviews_df.dropna(subset=["review_text"]).copy()
        reviews_clean["review_text"] = reviews_clean["review_text"].astype(str).str.strip()

        agg_reviews = (
            reviews_clean.groupby("product_id")["review_text"]
            .agg(
                review_count="count",
                review_text=lambda s: str(s.iloc[0]) if len(s) > 0 else "",
                reviews_combined=lambda s: " ".join(s.head(5).tolist()),
                review_skin_mentions=lambda s: {
                    kw: int(s.str.lower().str.contains(rf"\b{kw}\b", regex=True).sum())
                    for kw in ["oily", "dry", "sensitive", "combination", "acne"]
                },
            )
            .reset_index()
        )
        print(f"[INFO] Aggregated {len(reviews_clean)} reviews across {len(agg_reviews)} products.")
        products_df = pd.merge(products_df, agg_reviews, on="product_id", how="left")
        products_df["review_count"] = products_df["review_count"].fillna(0).astype(int)

    return products_df


def _extract_price(val):
    """Extract numeric price from strings like '$42', '$24.99'."""
    if pd.isna(val):
        return np.nan
    nums = re.findall(r"\d+\.?\d*", str(val))
    return float(nums[0]) if nums else np.nan


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[INFO] Cleaning DataFrame. Shape before: {df.shape}")

    df = df.dropna(how="all")

    dedup_cols = [c for c in ["product_id", "product_name", "brand"] if c in df.columns]
    if "product_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["product_id"])
        print(f"  Dropped {before - len(df)} duplicate rows (on product_id)")
    elif dedup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols)
        print(f"  Dropped {before - len(df)} duplicate rows (on {dedup_cols})")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    if "price" in df.columns:
        df["price_numeric"] = df["price"].apply(_extract_price)
        print(f"  Price range: ${df['price_numeric'].min():.2f} - ${df['price_numeric'].max():.2f}")

    if "category" in df.columns:
        df["category"] = df["category"].fillna("Skincare").str.strip()

    if "brand" in df.columns:
        df["brand"] = df["brand"].fillna("Unknown").str.strip()

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Determine text source for TF-IDF: use ingredients if present, else product_description
    if "ingredients" not in df.columns or df["ingredients"].isna().all():
        if "product_description" in df.columns:
            print("  [INFO] No separate 'ingredients' column; using 'product_description' for TF-IDF text.")
            df["ingredients"] = df["product_description"].fillna("")

    before = len(df)
    df = df[df["ingredients"].notna()]
    df = df[df["ingredients"].str.len() > 10]
    print(f"  Dropped {before - len(df)} rows with missing/short description/ingredients")

    df = df.reset_index(drop=True)
    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


def preprocess_text(text: str) -> str:
    """Normalize product description / ingredient text for TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tfidf(df: pd.DataFrame):
    """Fit a TF-IDF vectorizer on preprocessed product text."""
    print("\n[INFO] Building TF-IDF vectors on product descriptions & ingredients...")

    combined = (
        df["product_name"].fillna("") + " " +
        df["category"].fillna("") + " " +
        df["ingredients"].fillna("")
    )
    df["ingredients_clean"] = combined.apply(preprocess_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        stop_words="english",
        strip_accents="unicode",
        analyzer="word",
    )

    feature_matrix = vectorizer.fit_transform(df["ingredients_clean"])
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature matrix shape: {feature_matrix.shape}")

    return vectorizer, feature_matrix


def plot_eda(df: pd.DataFrame):
    n_plots = sum([
        "category" in df.columns,
        "price_numeric" in df.columns,
        "brand" in df.columns,
    ])
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Korean Skincare Products — EDA", fontsize=13, fontweight="bold")
    ax_idx = 0

    if "category" in df.columns:
        cat_counts = df["category"].value_counts().head(12)
        axes[ax_idx].barh(cat_counts.index[::-1], cat_counts.values[::-1], color="#4FC3F7")
        axes[ax_idx].set_title("Product Categories")
        axes[ax_idx].set_xlabel("Count")
        ax_idx += 1

    if "price_numeric" in df.columns:
        valid_prices = df["price_numeric"].dropna()
        axes[ax_idx].hist(valid_prices, bins=25, color="#81C784", edgecolor="black", linewidth=0.5)
        axes[ax_idx].set_title("Price Distribution")
        axes[ax_idx].set_xlabel("Price (USD)")
        axes[ax_idx].set_ylabel("Count")
        ax_idx += 1

    if "brand" in df.columns:
        top_brands = df["brand"].value_counts().head(10)
        axes[ax_idx].barh(top_brands.index[::-1], top_brands.values[::-1], color="#CE93D8")
        axes[ax_idx].set_title("Top 10 Brands")
        axes[ax_idx].set_xlabel("Count")

    plt.tight_layout()
    out_path = EVAL_DIR / "products_eda.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] EDA chart -> {out_path}")


def save_artifacts(df, vectorizer, feature_matrix):
    with open(MODELS_DIR / "products_clean.pkl", "wb") as f:
        pickle.dump(df, f)
    print(f"[SAVED] Clean products DataFrame -> {MODELS_DIR / 'products_clean.pkl'}")

    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[SAVED] TF-IDF vectorizer -> {MODELS_DIR / 'tfidf_vectorizer.pkl'}")

    with open(MODELS_DIR / "product_features.pkl", "wb") as f:
        pickle.dump(feature_matrix, f)
    print(f"[SAVED] Product feature matrix -> {MODELS_DIR / 'product_features.pkl'}")

    stats = {
        "total_products": int(len(df)),
        "total_reviews_linked": int(df["review_count"].sum()) if "review_count" in df.columns else 0,
        "columns": list(df.columns),
        "categories": df["category"].value_counts().to_dict() if "category" in df.columns else {},
        "price_range": {
            "min": float(df["price_numeric"].min()) if "price_numeric" in df.columns else None,
            "max": float(df["price_numeric"].max()) if "price_numeric" in df.columns else None,
            "mean": round(float(df["price_numeric"].mean()), 2) if "price_numeric" in df.columns else None,
        },
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
    }
    with open(PRODUCTS_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVED] Dataset stats -> {PRODUCTS_DIR / 'dataset_stats.json'}")


def main():
    print("=" * 60)
    print(" Korean Skincare Products Preprocessing — Stage 4")
    print("=" * 60)

    csvs = discover_csvs(PRODUCTS_DIR)
    dfs  = load_and_audit(csvs)
    df   = merge_dataframes(dfs)
    df   = clean_dataframe(df)
    vectorizer, feature_matrix = build_tfidf(df)
    plot_eda(df)
    save_artifacts(df, vectorizer, feature_matrix)

    print("\n[DONE] Korean products preprocessing complete.")
    print(f"       Products:  {len(df)}")
    print(f"       Features:  {feature_matrix.shape[1]}")


if __name__ == "__main__":
    main()
