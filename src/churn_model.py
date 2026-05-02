"""
src/churn_model.py
------------------
Churn Prediction Pipeline
  - Target: is_churned (no order in 60 days)
  - Models: Logistic Regression + Random Forest (comparison)
  - Outputs: probabilities, feature importance, evaluation charts
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from sklearn.pipeline        import Pipeline
from sklearn.calibration     import CalibratedClassifierCV

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection
# ─────────────────────────────────────────────────────────────────────────────
CHURN_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "discount_usage_rate",
    "avg_discount_pct",
    "orders_per_month",
    "lifespan_days",
    "recent_spend_ratio",
    "mayo_spend_share",
    "sauces_spend_share",
    "dips_spend_share",
    "distinct_products_bought",
    "tenure_days",
]

CATEGORICAL_FEATURES = ["city", "channel"]


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_churn_data(df: pd.DataFrame):
    """
    Returns X (features), y (target), feature_names list.
    Handles categorical encoding and missing values.
    """
    df = df.copy()

    # encode categoricals
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    enc_cols = [c + "_enc" for c in cat_cols]
    feature_cols = [f for f in CHURN_FEATURES if f in df.columns] + enc_cols

    X = df[feature_cols].fillna(0).values
    y = df["is_churned"].values.astype(int)

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    """Trains Logistic Regression, Random Forest, Gradient Boosting."""
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5, max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=4,
            subsample=0.8, random_state=42
        ),
    }

    fitted = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_tr_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model

    return fitted, scaler


def cross_val_comparison(models, scaler, X, y):
    """5-fold CV AUC comparison across models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        X_use = scaler.transform(X) if name == "Logistic Regression" else X
        scores = cross_val_score(model, X_use, y, scoring="roc_auc", cv=cv)
        results[name] = {"mean_auc": scores.mean(), "std_auc": scores.std()}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, scaler, X_test, y_test, model_name: str,
                   feature_names: list, verbose: bool = True):
    """Full evaluation report for one model."""
    X_use  = scaler.transform(X_test) if model_name == "Logistic Regression" else X_test
    y_pred = model.predict(X_use)
    y_prob = model.predict_proba(X_use)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    if verbose:
        print(f"\n── {model_name} ──")
        print(f"   ROC-AUC : {auc:.4f}")
        print(f"   Avg Prec: {ap:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Active","Churned"]))

    result = {
        "model_name":    model_name,
        "y_pred":        y_pred,
        "y_prob":        y_prob,
        "auc":           auc,
        "avg_precision": ap,
        "feature_names": feature_names,
    }

    # Feature importance
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        result["feature_importance"] = fi
    elif hasattr(model, "coef_"):
        fi = pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
        result["feature_importance"] = fi

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(eval_results: list, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ECC71", "#3498DB", "#E74C3C"]
    for res, col in zip(eval_results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=col,
                label=f"{res['model_name']} (AUC={res['auc']:.3f})")
    ax.plot([0,1],[0,1],"--",color="#BDC3C7", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Churn Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "05_roc_curves.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '05_roc_curves.png'}")
    return fig


def plot_feature_importance(eval_result: dict, top_n: int = 15, save: bool = True):
    fi = eval_result.get("feature_importance")
    if fi is None:
        return None
    fi_top = fi.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2ECC71" if v > fi_top.median() else "#3498DB" for v in fi_top.values]
    ax.barh(fi_top.index[::-1], fi_top.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(
        f"Top {top_n} Features — {eval_result['model_name']}\nChurn Prediction",
        fontsize=13, fontweight="bold"
    )
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fname = f"06_feature_importance_{eval_result['model_name'].replace(' ','_')}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / fname}")
    return fig


def plot_confusion_matrix(eval_result: dict, y_test, save: bool = True):
    cm = confusion_matrix(y_test, eval_result["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Active","Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {eval_result['model_name']}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fname = f"07_confusion_{eval_result['model_name'].replace(' ','_')}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / fname}")
    return fig


def plot_churn_risk_distribution(churn_df: pd.DataFrame, save: bool = True):
    """Histogram of churn probability scores coloured by actual label."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#2ECC71", "Active"), (1, "#E74C3C", "Churned")]:
        subset = churn_df[churn_df["is_churned"] == label]["churn_prob"]
        ax.hist(subset, bins=40, alpha=0.65, color=color, label=name, edgecolor="white")
    ax.axvline(0.5, color="#2C3E50", lw=2, ls="--", label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted Churn Probability", fontsize=11)
    ax.set_ylabel("Number of Customers", fontsize=11)
    ax.set_title("Churn Risk Score Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "08_churn_risk_distribution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {OUT_DIR / '08_churn_risk_distribution.png'}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Master function
# ─────────────────────────────────────────────────────────────────────────────

def run_churn_model(ml_features: pd.DataFrame, verbose: bool = True):
    """
    Full churn prediction pipeline.
    Returns enriched df with churn_prob and churn_risk_tier columns.
    """
    print("🔴  Running churn prediction model...")

    X, y, feature_names = prepare_churn_data(ml_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # train
    models, scaler = train_models(X_train, y_train)

    # CV comparison
    cv_res = cross_val_comparison(models, scaler, X, y)
    if verbose:
        print("\n  5-Fold CV AUC Comparison:")
        for mname, res in cv_res.items():
            print(f"    {mname:<25}: {res['mean_auc']:.4f} ± {res['std_auc']:.4f}")

    # evaluate all
    eval_results = []
    for mname, model in models.items():
        res = evaluate_model(model, scaler, X_test, y_test, mname, feature_names, verbose=verbose)
        eval_results.append(res)

    # pick best by AUC
    best_result = max(eval_results, key=lambda r: r["auc"])
    best_model  = models[best_result["model_name"]]
    best_name   = best_result["model_name"]
    if verbose:
        print(f"\n  ⭐ Best model: {best_name}  (AUC={best_result['auc']:.4f})")

    # score all customers
    X_all  = scaler.transform(X) if best_name == "Logistic Regression" else X
    probs  = best_model.predict_proba(X_all)[:, 1]

    df_out = ml_features.copy()
    df_out["churn_prob"] = probs
    df_out["churn_risk_tier"] = pd.cut(
        probs,
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    # charts
    plot_roc_curves(eval_results, y_test)
    plot_feature_importance(best_result)
    plot_confusion_matrix(best_result, y_test)
    plot_churn_risk_distribution(df_out)

    print("  ✅  Churn model complete.\n")
    return df_out, best_model, scaler, best_result
