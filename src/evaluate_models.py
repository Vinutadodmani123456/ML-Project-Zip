"""
============================================================
Facial Skin Condition Detection
Phase 5 — Model Evaluation
============================================================

Loads trained model results and generates:
  1. Comparison table (console + reports/model_comparison.png)
  2. Confusion matrices for all 7 models (reports/confusion_matrices.png)
  3. Detailed classification report for the best model

Functions
---------
compare_models()             → prints table + saves bar chart
plot_confusion_matrices()    → saves confusion matrix grid
generate_classification_report() → prints best model report

Run standalone:
    python src/evaluate_models.py

Author: MCA Project
"""

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

PALETTE = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB", "#9B59B6", "#1ABC9C"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_results() -> dict:
    pkl_path = MODELS_DIR / "model_results.pkl"
    if not pkl_path.exists():
        print(f"\n[ERROR] model_results.pkl not found: {pkl_path}")
        print("        Run python src/train_models.py first.")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ── Core Functions ────────────────────────────────────────────────────────────

def compare_models(payload: dict | None = None) -> pd.DataFrame:
    """
    Print a ranked comparison table and save reports/model_comparison.png.

    Parameters
    ----------
    payload : dict, optional
        Pre-loaded results dict. If None, loaded from models/model_results.pkl.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by CV accuracy (descending).
    """
    if payload is None:
        payload = _load_results()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_results = payload["model_results"]
    best_name     = payload["best_model_name"]

    rows = []
    for name, res in model_results.items():
        rows.append({
            "Model":        name,
            "CV Mean Acc":  res["cv_mean"],
            "CV Std":       res["cv_std"],
            "Test Acc":     res["test_accuracy"],
        })

    df = pd.DataFrame(rows).sort_values("CV Mean Acc", ascending=False).reset_index(drop=True)
    df.index += 1   # 1-indexed rank

    # ── Console table ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  MODEL COMPARISON — Ranked by Cross-Validation Accuracy")
    print("=" * 68)
    print(f"\n  {'Rank':<6} {'Model':<24} {'CV Acc':<14} {'CV Std':<12} {'Test Acc'}")
    print("  " + "-" * 60)
    for rank, row in df.iterrows():
        marker = " ◀ BEST" if row["Model"] == best_name else ""
        print(f"  {rank:<6} {row['Model']:<24} {row['CV Mean Acc']:.4f}         "
              f"{row['CV Std']:.4f}     {row['Test Acc']:.4f}{marker}")
    print("  " + "-" * 60)

    # ── Bar chart ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ML Model Comparison — Facial Skin Condition Detection",
                 fontsize=15, fontweight="bold", y=1.02)

    models_sorted = df["Model"].tolist()
    cv_accs       = df["CV Mean Acc"].tolist()
    cv_stds       = df["CV Std"].tolist()
    test_accs     = df["Test Acc"].tolist()
    colours       = PALETTE[:len(models_sorted)]

    # CV accuracy bar chart
    bars = axes[0].barh(models_sorted[::-1], cv_accs[::-1],
                        color=colours[::-1], edgecolor="white", linewidth=0.5)
    axes[0].errorbar(cv_accs[::-1],
                     range(len(models_sorted)),
                     xerr=cv_stds[::-1],
                     fmt="none", color="black", capsize=4, linewidth=1.5)
    axes[0].set_xlabel("Accuracy", fontsize=12)
    axes[0].set_title("5-Fold CV Accuracy (with ± std)", fontsize=12)
    axes[0].set_xlim(0, 1.05)
    for bar, val in zip(bars, cv_accs[::-1]):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=10)

    # Test accuracy bar chart
    bars2 = axes[1].barh(models_sorted[::-1], test_accs[::-1],
                          color=colours[::-1], edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Accuracy", fontsize=12)
    axes[1].set_title("Hold-out Test Set Accuracy", fontsize=12)
    axes[1].set_xlim(0, 1.05)
    for bar, val in zip(bars2, test_accs[::-1]):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    save_path = REPORTS_DIR / "model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] {save_path}")

    return df


def plot_confusion_matrices(payload: dict | None = None) -> None:
    """
    Plot confusion matrices for all models in a grid and save to
    reports/confusion_matrices.png.

    Parameters
    ----------
    payload : dict, optional
        Pre-loaded results dict. If None, loaded from models/model_results.pkl.
    """
    if payload is None:
        payload = _load_results()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_results = payload["model_results"]
    y_test        = payload["y_test"]
    le            = payload["label_encoder"]
    class_names   = le.classes_

    n_models = len(model_results)
    n_cols   = 3 if n_models > 4 else 2
    n_rows   = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    fig.suptitle("Confusion Matrices — All Models",
                 fontsize=16, fontweight="bold", y=1.01)

    for idx, (name, res) in enumerate(model_results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        disp.plot(ax=axes[idx], colorbar=False, cmap="Blues")
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
        axes[idx].set_yticklabels(class_names, fontsize=8)
        axes[idx].set_xlabel("Predicted", fontsize=9)
        axes[idx].set_ylabel("Actual", fontsize=9)

    # Hide any unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_path = REPORTS_DIR / "confusion_matrices.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {save_path}")


def generate_classification_report(payload: dict | None = None) -> None:
    """
    Print the detailed per-class classification report for the best model.

    Parameters
    ----------
    payload : dict, optional
        Pre-loaded results dict. If None, loaded from models/model_results.pkl.
    """
    if payload is None:
        payload = _load_results()

    best_name     = payload["best_model_name"]
    model_results = payload["model_results"]
    y_test        = payload["y_test"]
    le            = payload["label_encoder"]
    y_pred        = model_results[best_name]["y_pred"]

    print(f"\n── Classification Report — {best_name} (Best Model) ──────────")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                zero_division=0))


def plot_feature_importance(payload: dict | None = None) -> None:
    """
    If the best model supports feature importances (Random Forest / XGBoost),
    plot and save the top 20 features to reports/feature_importance.png.
    """
    if payload is None:
        payload = _load_results()

    best_model    = payload["best_model"]
    feature_cols  = payload.get("feature_cols", [])
    best_name     = payload["best_model_name"]

    if not hasattr(best_model, "feature_importances_"):
        print(f"\n  [INFO] {best_name} does not expose feature importances — skipping.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    importances = best_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]
    top_feats   = [feature_cols[i] if i < len(feature_cols) else f"F{i}" for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_feats[::-1], top_vals[::-1], color="#3498DB", edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top 20 Feature Importances — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {save_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    """Run the full evaluation pipeline."""
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION — Generating Reports & Charts")
    print("=" * 60)

    payload = _load_results()

    compare_models(payload)
    plot_confusion_matrices(payload)
    generate_classification_report(payload)
    plot_feature_importance(payload)

    print("\n  All evaluation artefacts saved to reports/")
    print("  Next step → cd app && python app.py")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
