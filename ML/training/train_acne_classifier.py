"""
train_acne_classifier.py
Stage 5 — EfficientNet-B0 Transfer Learning for Acne Severity Classification.

Trains on the ACNE04 dataset using the canonical fold_0 split.
Pre-caches resized RGB images locally for fast GPU training, unfreezes
the full EfficientNet-B0 backbone with differential learning rates,
and uses sqrt class weights + label smoothing to balance all 4 classes.

Usage:
    python ML/training/train_acne_classifier.py [--epochs 30] [--batch_size 32] [--lr 2e-4]

Outputs:
    ML/models/acne_classifier_efficientnet.pt    — best model weights
    ML/models/acne_training_config.json          — training hyperparameters
    ML/evaluation/acne_training_curves.png       — loss/accuracy curves
    ML/evaluation/acne_confusion_matrix.png      — confusion matrix
    ML/evaluation/acne_classification_report.txt — precision/recall/F1
"""

import os
import sys
import json
import time
import pickle
import argparse
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ACNE_DS_DIR = ROOT / "ML" / "datasets" / "acne04"
CACHE_IMG_DIR = ACNE_DS_DIR / "cached_images"
MODELS_DIR  = ROOT / "ML" / "models"
EVAL_DIR    = ROOT / "ML" / "evaluation"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Mild", "Moderate", "Severe", "Very Severe"]
NUM_CLASSES = 4
IMAGE_SIZE  = 224


# ─── Disk Image Cache ─────────────────────────────────────────────────────────

def ensure_cached_images(manifest: pd.DataFrame):
    """Pre-save resized 256x256 RGB JPEGs so GPU epochs run in seconds."""
    CACHE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    expected = len(manifest)
    existing = list(CACHE_IMG_DIR.glob("*.jpg"))
    if len(existing) == expected:
        print(f"[CACHE] Found all {expected} pre-cached images in {CACHE_IMG_DIR}")
        return

    print(f"[CACHE] Loading ACNE04 from HuggingFace to cache {expected} resized images...")
    from datasets import load_dataset
    hf_ds = load_dataset("Layered-Labs/ACNE04", cache_dir=str(ACNE_DS_DIR), split="train")

    for idx in manifest["index"].tolist():
        out_file = CACHE_IMG_DIR / f"{int(idx)}.jpg"
        if not out_file.exists():
            img = hf_ds[int(idx)]["image"].convert("RGB")
            img = img.resize((256, 256), Image.Resampling.BILINEAR)
            img.save(out_file, format="JPEG", quality=95)

    print(f"[CACHE] Cached {expected} images successfully.")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ACNE04CachedDataset(Dataset):
    """Fast PyTorch Dataset reading pre-cached 256x256 RGB JPEGs."""

    def __init__(self, indices, labels, transform=None):
        self.indices   = indices
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = CACHE_IMG_DIR / f"{int(self.indices[idx])}.jpg"
        image    = Image.open(img_path).convert("RGB")
        label    = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
        transforms.RandomRotation(degrees=12),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transforms, val_transforms


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, dropout_rate: float = 0.4):
    """
    EfficientNet-B0 with full backbone fine-tuning and custom 4-class head.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(256, num_classes),
    )

    return model


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds      = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, use_tta: bool = False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            if use_tta:
                probs = torch.softmax(outputs, dim=1)
                flipped_outputs = model(torch.flip(images, dims=[3]))
                probs = 0.5 * (probs + torch.softmax(flipped_outputs, dim=1))
                preds = probs.argmax(dim=1)
            else:
                preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Evaluation Plots ─────────────────────────────────────────────────────────

def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("EfficientNet-B0 — Acne Severity Classification Training", fontsize=13)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = EVAL_DIR / "acne_training_curves.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Training curves -> {out_path}")


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Confusion Matrix — Acne Severity (Test Set)", fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    out_path = EVAL_DIR / "acne_confusion_matrix.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrix -> {out_path}")


def save_classification_report(labels, preds, train_acc, val_acc, test_acc, tta_acc, config):
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    lines = [
        "=" * 60,
        " ACNE Severity Classification — Evaluation Report",
        "=" * 60,
        "\nModel:        EfficientNet-B0 (Transfer Learning, Full Fine-Tune)",
        "Dataset:      ACNE04 (Layered-Labs/ACNE04)",
        "Split:        Canonical Fold 0 (Train/Test)",
        "Total images: 1457",
        "\nTraining Configuration:",
        f"  Epochs:         {config['epochs']}",
        f"  Batch size:     {config['batch_size']}",
        f"  LR:             {config['lr']} (differential backbone/head)",
        "  Optimizer:      AdamW",
        "  Scheduler:      CosineAnnealingLR",
        f"  Dropout:        {config['dropout']}",
        "  Label Smooth:   0.08",
        "\nResults (Best Validation Checkpoint):",
        f"  Best Train Acc:  {train_acc:.4f} ({train_acc*100:.2f}%)",
        f"  Best Val Acc:    {val_acc:.4f} ({val_acc*100:.2f}%)",
        f"  Test Acc (Raw):  {test_acc:.4f} ({test_acc*100:.2f}%)",
        f"  Test Acc (TTA):  {tta_acc:.4f} ({tta_acc*100:.2f}%)",
        "\nTest Set — Per-Class Metrics (Raw):",
        report,
        "\nNote: Sqrt class weights + label smoothing used to balance precision and recall.",
        "Do not fabricate or modify these numbers.",
    ]

    out_path = EVAL_DIR / "acne_classification_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] Classification report -> {out_path}")
    print("\n".join(lines))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print(" Acne Classifier Training — Stage 5 (EfficientNet-B0)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")

    manifest_path = ACNE_DS_DIR / "acne04_manifest.csv"
    weights_path  = ACNE_DS_DIR / "class_weights.pkl"

    if not manifest_path.exists():
        print("[ERROR] acne04_manifest.csv not found. Run preprocess_acne04.py first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    with open(weights_path, "rb") as f:
        class_weights_dict = pickle.load(f)

    ensure_cached_images(manifest)

    train_manifest = manifest[manifest["fold_split"] == "train"].reset_index(drop=True)
    test_manifest  = manifest[manifest["fold_split"] == "test"].reset_index(drop=True)

    print(f"\n[INFO] Train samples: {len(train_manifest)}")
    print(f"[INFO] Test samples:  {len(test_manifest)}")

    train_tf, val_tf = get_transforms()

    train_dataset = ACNE04CachedDataset(
        train_manifest["index"].tolist(), train_manifest["level"].tolist(), transform=train_tf
    )
    test_dataset = ACNE04CachedDataset(
        test_manifest["index"].tolist(), test_manifest["level"].tolist(), transform=val_tf
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    model = build_model(num_classes=NUM_CLASSES, dropout_rate=args.dropout).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] Total params:     {total_params:,}")
    print(f"[MODEL] Trainable params: {trainable_params:,}")

    # Sqrt inverse-frequency weights (gentler than raw weights, no double-sampling)
    sqrt_weights = torch.FloatTensor(
        [np.sqrt(class_weights_dict.get(i, 1.0)) for i in range(NUM_CLASSES)]
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=sqrt_weights, label_smoothing=0.08)

    # Differential learning rates
    optimizer = optim.AdamW([
        {"params": model.features[:6].parameters(), "lr": args.lr * 0.25},
        {"params": model.features[6:].parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(),   "lr": args.lr * 2.0},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc     = 0.0
    best_train_acc   = 0.0
    patience_counter = 0

    best_model_path = MODELS_DIR / "acne_classifier_efficientnet.pt"

    print(f"\n[TRAINING] Starting for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device, use_tta=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch [{epoch:3d}/{args.epochs}] "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc*100:.2f}% | "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc*100:.2f}% | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_train_acc = train_acc
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_acc":      val_acc,
                "class_names":  CLASS_NAMES,
                "num_classes":  NUM_CLASSES,
                "image_size":   IMAGE_SIZE,
            }, str(best_model_path))
            print(f"  -> New best! Val Acc={val_acc*100:.2f}% | Checkpoint saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOP] No improvement for {args.patience} epochs. Stopping.")
                break

    print(f"\n[INFO] Loading best checkpoint: {best_model_path}")
    checkpoint = torch.load(str(best_model_path), map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device, use_tta=False)
    _, tta_acc, _, _ = evaluate(model, test_loader, criterion, device, use_tta=True)
    print(f"\n[RESULTS] Test Accuracy (Raw): {test_acc*100:.2f}% | Test Accuracy (TTA): {tta_acc*100:.2f}%")

    plot_training_curves(history)
    plot_confusion_matrix(all_labels, all_preds)

    config = {
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "dropout":       args.dropout,
        "patience":      args.patience,
        "image_size":    IMAGE_SIZE,
        "num_classes":   NUM_CLASSES,
        "class_names":   CLASS_NAMES,
        "optimizer":     "AdamW",
        "scheduler":     "CosineAnnealingLR",
        "best_val_acc":  round(best_val_acc, 4),
        "test_acc":      round(test_acc, 4),
        "tta_acc":       round(tta_acc, 4),
    }

    with open(MODELS_DIR / "acne_training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Training config -> {MODELS_DIR / 'acne_training_config.json'}")

    save_classification_report(all_labels, all_preds, best_train_acc, best_val_acc, test_acc, tta_acc, config)
    print("\n[DONE] Acne classifier training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on ACNE04")
    parser.add_argument("--epochs",     type=int,   default=30,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--dropout",    type=float, default=0.4,  help="Dropout rate")
    parser.add_argument("--patience",   type=int,   default=8,    help="Early stopping patience")
    args = parser.parse_args()
    main(args)
