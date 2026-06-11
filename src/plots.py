"""
plots.py
--------
Publication-quality figure helpers for the student dropout project.
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EXPLORATION_DIR, FIGURES_DIR
from src.utils import ensure_dirs, save_figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
DROPOUT_PALETTE = {0: "#2ecc71", 1: "#e74c3c"}


def _save(fig: plt.Figure, directory: str, fname: str) -> None:
    ensure_dirs(directory)
    save_figure(fig, os.path.join(directory, fname))


def plot_class_balance_binary(y: pd.Series, save: bool = True) -> plt.Figure:
    """Bar chart of binary class distribution."""
    counts = y.value_counts().rename({1: "Dropout", 0: "Graduate"})
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="white",
        width=0.5,
    )
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Binary Class Distribution (Dropout vs. Graduate)")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    if save:
        _save(fig, FIGURES_DIR, "class_balance_binary.png")
    return fig


def plot_feature_importance_top(
    importance_df: pd.DataFrame,
    score_col: str,
    title: str,
    n: int = 10,
    save: bool = True,
    fname: str = "feature_importance_top10.png",
) -> plt.Figure:
    """Horizontal bar chart for the top-n ranked features."""
    top = importance_df.nlargest(n, score_col).sort_values(score_col)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top[score_col], color="#3498db", edgecolor="white")
    ax.set_xlabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    if save:
        _save(fig, FIGURES_DIR, fname)
    return fig


def plot_group_importance(
    group_summary: pd.DataFrame,
    score_col: str = "mean_score",
    save: bool = True,
) -> plt.Figure:
    """Bar chart comparing mean feature importance by group."""
    df = group_summary.sort_values(score_col, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        df["group"],
        df[score_col],
        color=sns.color_palette("muted", len(df)),
        edgecolor="white",
    )
    ax.set_ylabel("Mean Importance Score")
    ax.set_xlabel("Feature Group")
    ax.set_title("Mean Feature Importance by Group")
    fig.tight_layout()
    if save:
        _save(fig, FIGURES_DIR, "group_importance.png")
    return fig


def plot_model_comparison(
    cv_results: pd.DataFrame,
    metric: str = "accuracy",
    save: bool = True,
) -> plt.Figure:
    """Bar chart comparing models by a given cross-validation metric."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    df = cv_results.sort_values(mean_col, ascending=False).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        df["model"],
        df[mean_col],
        yerr=df[std_col],
        color=sns.color_palette("muted", len(df)),
        edgecolor="white",
        capsize=4,
        error_kw={"elinewidth": 1.2},
    )
    ax.bar_label(bars, fmt="%.3f", padding=6, fontsize=9)
    ax.set_ylabel(f"Mean {metric.replace('_', ' ').title()}")
    ax.set_title(f"10-Fold CV - Model Comparison ({metric.replace('_', ' ').title()})")
    ax.set_ylim(0, df[mean_col].max() * 1.15)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    if save:
        _save(fig, FIGURES_DIR, f"model_{metric}_comparison.png")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save: bool = True,
) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    labels = ["Graduate (0)", "Dropout (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    if save:
        _save(fig, FIGURES_DIR, "confusion_matrix_best_model.png")
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    model_name: str,
    save: bool = True,
) -> plt.Figure:
    """ROC curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save:
        _save(fig, FIGURES_DIR, "roc_curve_best_model.png")
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    avg_precision: float,
    model_name: str,
    fname: str = "precision_recall_best_model.png",
    baseline: float = 0.39,
) -> plt.Figure:
    """Precision-recall curve with average precision annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(
        recall,
        precision,
        where="post",
        color="#e74c3c",
        lw=2,
        label=f"{model_name} (AP = {avg_precision:.3f})",
    )
    ax.axhline(
        baseline,
        color="gray",
        ls="--",
        lw=1,
        label=f"Baseline (random, {baseline:.0%})",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve - Best Model", fontsize=13)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend()
    fig.tight_layout()
    _save(fig, FIGURES_DIR, fname)
    return fig


def plot_lr_coefficients(
    coefs: pd.Series,
    fname: str = "lr_coefficients.png",
) -> plt.Figure:
    """Horizontal bar chart of standardized logistic regression coefficients."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if coef > 0 else "#3498db" for coef in coefs]
    coefs.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardized Coefficient (positive = higher dropout risk)")
    ax.set_title("Logistic Regression Coefficients - Top Features")
    fig.tight_layout()
    _save(fig, FIGURES_DIR, fname)
    return fig


def plot_cv_boxplot(
    fold_scores_dict: dict,
    metric: str = "f1",
    fname: str | None = None,
) -> plt.Figure:
    """Horizontal box plot for per-fold model scores."""
    if not fold_scores_dict:
        raise ValueError("fold_scores_dict must not be empty")

    first_value = next(iter(fold_scores_dict.values()))
    if isinstance(first_value, dict):
        plot_source = {
            model: scores[metric]
            for model, scores in fold_scores_dict.items()
        }
    else:
        plot_source = fold_scores_dict

    df = pd.DataFrame(plot_source).T
    df["median"] = df.median(axis=1)
    df = df.sort_values("median")
    plot_data = df.drop(columns="median")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        plot_data.values.T,
        vert=False,
        patch_artist=True,
        labels=plot_data.index,
        boxprops=dict(facecolor="#3498db", alpha=0.6),
        medianprops=dict(color="#1f1f1f", linewidth=1.5),
    )
    ax.set_xlabel(f"{metric.upper()} Score (10-fold CV)")
    ax.set_title(f"Cross-Validation Stability - {metric.upper()}")
    ax.axvline(plot_data.values.max(), color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    _save(fig, FIGURES_DIR, fname or f"cv_boxplot_{metric}.png")
    return fig


def plot_feature_subset_comparison(
    df_all: pd.DataFrame,
    df_top10: pd.DataFrame,
    metric: str = "f1_mean",
    fname: str = "all_vs_top10_comparison.png",
) -> plt.Figure:
    """Grouped horizontal bar chart comparing all features vs. top-10 features."""
    models = df_all.sort_values(metric, ascending=False)["model"].tolist()
    all_scores = df_all.set_index("model")[metric].reindex(models).values
    top10_scores = df_top10.set_index("model")[metric].reindex(models).values

    y_pos = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        y_pos + height / 2,
        all_scores,
        height,
        label="All 36 Features",
        color="#3498db",
        alpha=0.85,
    )
    ax.barh(
        y_pos - height / 2,
        top10_scores,
        height,
        label="Top-10 Features",
        color="#e67e22",
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel(f"{metric.replace('_mean', '').upper()} Score")
    ax.set_title("All Features vs. Top-10 Features - Model Comparison")
    ax.set_xlim([0.7, 1.0])
    ax.legend()
    fig.tight_layout()
    _save(fig, FIGURES_DIR, fname)
    return fig


def plot_dropout_rate_by_category(
    df: pd.DataFrame,
    binary_features: list[str],
    fname: str = "11_dropout_rate_by_category.png",
    save_dir: str | None = None,
) -> plt.Figure:
    """Plot dropout rate per value for key binary or categorical features."""
    label_map = {
        "Tuition fees up to date": {0: "Unpaid", 1: "Paid"},
        "Debtor": {0: "No Debt", 1: "Debtor"},
        "Scholarship holder": {0: "No Scholarship", 1: "Scholarship"},
        "Gender": {0: "Female", 1: "Male"},
        "Daytime/evening attendance": {0: "Evening", 1: "Daytime"},
        "Displaced": {0: "Not Displaced", 1: "Displaced"},
    }

    overall_rate = df["Target"].mean() * 100
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, feat in enumerate(binary_features):
        ax = axes[idx]
        rates = (df.groupby(feat)["Target"].mean() * 100).reindex([0, 1])
        counts = df.groupby(feat)["Target"].count().reindex([0, 1])
        labels = [label_map.get(feat, {}).get(val, str(val)) for val in rates.index]
        bars = ax.bar(labels, rates.values, color=["#3498db", "#e74c3c"], alpha=0.85, width=0.5)
        for bar, count in zip(bars, counts.fillna(0).astype(int).values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"n={count}",
                ha="center",
                fontsize=9,
            )
        ax.set_title(feat, fontsize=10, pad=6)
        ax.set_ylabel("Dropout Rate (%)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.axhline(overall_rate, color="gray", ls="--", lw=0.8, label=f"Overall {overall_rate:.0f}%")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Dropout Rate by Categorical Feature / Abbruchrate nach Kategorien", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir or EXPLORATION_DIR, fname)
    return fig


def plot_per_class_metrics(
    report_dict: dict,
    model_name: str,
    fname: str = "per_class_metrics.png",
) -> plt.Figure:
    """Grouped bar chart for per-class precision, recall, and F1."""
    classes = ["Graduate (0)", "Dropout (1)"]
    keys = ["0", "1"]
    metrics = ["precision", "recall", "f1-score"]
    x_pos = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#e74c3c"]
    for idx, (label, key, color) in enumerate(zip(classes, keys, colors)):
        values = [report_dict[key][metric] for metric in metrics]
        bars = ax.bar(x_pos + idx * width, values, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01),
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Class Metrics - {model_name}", fontsize=13)
    ax.legend()
    fig.tight_layout()
    _save(fig, FIGURES_DIR, fname)
    return fig


def plot_importance_consensus(
    mi_df: pd.DataFrame,
    fscore_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    n: int = 10,
    fname: str = "feature_importance_consensus.png",
) -> plt.Figure:
    """Rank-based heatmap comparing top feature positions across methods."""
    top_feats = mi_df.head(n)["feature"].tolist()
    mi_rank = {feature: rank for rank, feature in enumerate(mi_df["feature"], start=1)}
    f_rank = {feature: rank for rank, feature in enumerate(fscore_df["feature"], start=1)}
    rf_rank = {feature: rank for rank, feature in enumerate(rf_df["feature"], start=1)}

    ranks = pd.DataFrame({"Feature": top_feats})
    ranks["MI Rank"] = ranks["Feature"].map(mi_rank)
    ranks["F-score Rank"] = ranks["Feature"].map(f_rank).fillna(20).astype(int)
    ranks["RF Rank"] = ranks["Feature"].map(rf_rank).fillna(20).astype(int)
    ranks = ranks.set_index("Feature")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        ranks,
        annot=True,
        fmt="d",
        cmap="YlOrRd_r",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Rank (lower = better)"},
    )
    ax.set_title("Feature Importance Consensus - MI vs F-score vs RF", fontsize=13)
    fig.tight_layout()
    _save(fig, FIGURES_DIR, fname)
    return fig


def plot_top_features_violin(
    df: pd.DataFrame,
    features: list[str],
    fname: str = "12_top_features_violin.png",
) -> plt.Figure:
    """Violin plots for top academic features split by target."""
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 5), sharey=False)
    axes = np.atleast_1d(axes)
    plot_df = df.copy()
    plot_df["TargetLabel"] = plot_df["Target"].map({0: "Graduate", 1: "Dropout"})
    palette = {"Graduate": "#3498db", "Dropout": "#e74c3c"}

    for ax, feat in zip(axes, features):
        sns.violinplot(
            data=plot_df,
            x="TargetLabel",
            y=feat,
            hue="TargetLabel",
            ax=ax,
            palette=palette,
            inner="quartile",
            cut=0,
            legend=False,
        )
        ax.set_title(feat.replace("Curricular units ", "CU "), fontsize=9)
        ax.set_xlabel("")

    fig.suptitle("Top Academic Features by Dropout Status", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, EXPLORATION_DIR, fname)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
    fname: str = "13_correlation_heatmap_top10.png",
) -> plt.Figure:
    """Upper-triangular correlation heatmap for the selected features and target."""
    corr = df[features + ["Target"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        linewidths=0.5,
        square=True,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlation Heatmap - Top-10 Features + Target", fontsize=13)
    fig.tight_layout()
    _save(fig, EXPLORATION_DIR, fname)
    return fig


def plot_radar_model_comparison(
    cv_results: pd.DataFrame,
    fname: str = "radar_model_comparison.png",
) -> plt.Figure:
    """Radar chart comparing models across four core metrics."""
    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    count = len(labels)
    angles = [idx / float(count) * 2 * np.pi for idx in range(count)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for idx, (_, row) in enumerate(cv_results.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", lw=2, color=colors[idx % len(colors)], label=row["model"])
        ax.fill(angles, values, alpha=0.07, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.set_yticklabels(["75%", "80%", "85%", "90%", "95%"], fontsize=8)
    ax.set_title("Model Comparison - Radar Chart", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
    fig.tight_layout()
    _save(fig, FIGURES_DIR, fname)
    return fig
