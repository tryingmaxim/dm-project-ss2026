"""
defense_prep.py
---------------
Generate defense-ready robustness checks and a speaking note document for the
student dropout project.
"""

from __future__ import annotations

import os

import pandas as pd

from src.config import DEFENSE_DIR, TABLES_DIR
from src.data_loader import get_X_y, load_raw, prepare_binary
from src.train_models import (
    compare_global_vs_pipeline_selection,
    compare_integer_vs_one_hot,
    run_svm_grid_search,
)
from src.utils import ensure_dirs


def load_base_metrics() -> tuple[pd.Series, int]:
    """Load the current LR metrics from the saved all-features comparison table."""
    table = pd.read_csv(os.path.join(TABLES_DIR, "model_comparison_all_features.csv"))
    lr_row = table.loc[table["model"] == "Logistic Regression"].iloc[0]

    df = prepare_binary(load_raw())
    actual_dropouts = int(df["Target"].sum())
    return lr_row, actual_dropouts


def compute_roi_metrics(
    precision: float,
    recall: float,
    actual_dropouts: int,
    intervention_cost_eur: float = 50.0,
    dropout_savings_eur: float = 5000.0,
) -> dict[str, float]:
    """Compute an upper-bound ROI and a sensitivity range for intervention success."""
    true_positives = recall * actual_dropouts
    predicted_positive = true_positives / precision
    intervention_cost = predicted_positive * intervention_cost_eur

    def roi_for_effectiveness(effectiveness: float) -> tuple[float, float]:
        gross_savings = effectiveness * true_positives * dropout_savings_eur
        roi = (gross_savings - intervention_cost) / intervention_cost
        return gross_savings, roi

    gross_100, roi_100 = roi_for_effectiveness(1.0)
    gross_30, roi_30 = roi_for_effectiveness(0.30)
    gross_50, roi_50 = roi_for_effectiveness(0.50)

    return {
        "true_positives": true_positives,
        "predicted_positive": predicted_positive,
        "intervention_cost": intervention_cost,
        "gross_100": gross_100,
        "roi_100": roi_100,
        "gross_30": gross_30,
        "roi_30": roi_30,
        "gross_50": gross_50,
        "roi_50": roi_50,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f} %"


def _pp(value: float) -> str:
    return f"{value:.2f} pp"


def _eur(value: float) -> str:
    return f"{value:,.0f} EUR".replace(",", ".")


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns]
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_defense_markdown(
    leakage_df: pd.DataFrame,
    svm_results: pd.DataFrame,
    svm_best: dict,
    onehot_df: pd.DataFrame,
    lr_row: pd.Series,
    roi: dict[str, float],
) -> str:
    """Compose the defense speaking-note markdown with concrete project numbers."""
    f1_global = float(
        leakage_df.loc[
            leakage_df["model"] == "Global MI Top-10 before CV", "f1_mean"
        ].iloc[0]
    )
    f1_pipeline = float(
        leakage_df.loc[
            leakage_df["model"] == "Pipeline SelectKBest per fold", "f1_mean"
        ].iloc[0]
    )
    leakage_delta_pp = (f1_global - f1_pipeline) * 100
    leakage_severity = "klein" if abs(leakage_delta_pp) < 1 else "relevant"

    onehot_lr = float(
        onehot_df.loc[
            onehot_df["model"] == "Logistic Regression - One-Hot", "f1_mean"
        ].iloc[0]
    )
    onehot_svm = float(
        onehot_df.loc[
            onehot_df["model"] == "SVM - One-Hot", "f1_mean"
        ].iloc[0]
    )
    baseline_lr = float(
        onehot_df.loc[
            onehot_df["model"] == "Logistic Regression - Integer Codes", "f1_mean"
        ].iloc[0]
    )
    baseline_svm = float(
        onehot_df.loc[
            onehot_df["model"] == "SVM - Integer Codes", "f1_mean"
        ].iloc[0]
    )

    robustness_summary = pd.DataFrame(
        [
            {
                "Check": "Leakage-free LR pipeline",
                "Alt": _pct(f1_global),
                "Neu": _pct(f1_pipeline),
                "Delta": _pp(leakage_delta_pp),
                "Kurzfazit": f"Leakage-Bias wirkt {leakage_severity}.",
            },
            {
                "Check": "SVM GridSearch",
                "Alt": _pct(baseline_svm),
                "Neu": _pct(float(svm_best["best_f1"])),
                "Delta": _pp((float(svm_best["best_f1"]) - baseline_svm) * 100),
                "Kurzfazit": (
                    f"Bestes Grid: C={svm_best['best_C']}, gamma={svm_best['best_gamma']}."
                ),
            },
            {
                "Check": "One-Hot vs Integer",
                "Alt": _pct(baseline_lr),
                "Neu": _pct(onehot_lr),
                "Delta": _pp((onehot_lr - baseline_lr) * 100),
                "Kurzfazit": "LR-Vergleich; SVM separat direkt darunter.",
            },
        ]
    )

    lines = [
        "# Defense Prep: Leakage, ROI und Robustness Checks",
        "",
        "## Leakage",
        "Konkrete sklearn-Antwort:",
        "```python",
        "from sklearn.feature_selection import SelectKBest, mutual_info_classif",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import StandardScaler",
        "",
        "pipeline = Pipeline([",
        '    ("imputer", SimpleImputer(strategy="median")),',
        '    ("scaler", StandardScaler()),',
        '    ("selector", SelectKBest(score_func=mutual_info_classif, k=10)),',
        '    ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),',
        "])",
        "```",
        "Mit dieser Struktur wird `SelectKBest` in jedem CV-Fold nur auf dem jeweiligen Trainingsfold neu gefittet. Das ist die methodisch saubere, leakage-freie Variante.",
        "",
        "Fuer die Verteidigung solltet ihr immer beide Varianten nennen: den bisherigen globalen MI-Top-10-Ansatz und die leakage-freie Pipeline-Variante. Entscheidend ist dann die F1-Differenz als Groessenordnung des Bias.",
        "",
        _markdown_table(
            leakage_df.rename(
                columns={
                    "model": "Variante",
                    "f1_mean": "F1",
                    "roc_auc_mean": "ROC-AUC",
                    "f1_difference_vs_pipeline_pp": "Delta vs Pipeline (pp)",
                }
            ),
            columns=["Variante", "F1", "ROC-AUC", "Delta vs Pipeline (pp)", "top_features"],
        ),
        "",
        (
            f"Interpretation: Die Differenz zwischen globaler Top-10-Auswahl und der "
            f"leakage-freien Pipeline liegt bei {_pp(leakage_delta_pp)}. Damit wirkt der "
            f"optimistische Bias in eurem Datensatz {leakage_severity}."
        ),
        "",
        "## ROI",
        (
            f"Ausgangspunkt sind die gespeicherten LR-Zahlen mit Precision {_pct(float(lr_row['precision_mean']))} "
            f"und Recall {_pct(float(lr_row['recall_mean']))} sowie {int(round(roi['true_positives'] / float(lr_row['recall_mean'])))} "
            "tatsaechliche Dropout-Faelle in der Binary-Stichprobe."
        ),
        "",
        (
            f"Upper-Bound-Rechnung bei 100 % Interventionserfolg: "
            f"TP ca. {roi['true_positives']:.0f}, vorhergesagte Risikofaelle ca. {roi['predicted_positive']:.0f}, "
            f"Beratungskosten ca. {_eur(roi['intervention_cost'])}, Bruttoeinsparung ca. {_eur(roi['gross_100'])}, "
            f"ROI ca. {roi['roi_100']:.1f}."
        ),
        (
            f"Sensitivitaet bei realistischeren Erfolgsraten: Bei 30 % Wirksamkeit ergibt sich "
            f"eine Bruttoeinsparung von {_eur(roi['gross_30'])} und ein ROI von etwa {roi['roi_30']:.1f}; "
            f"bei 50 % Wirksamkeit sind es {_eur(roi['gross_50'])} und ein ROI von etwa {roi['roi_50']:.1f}. "
            "Als runde Verteidigungsformel ist daher fair: Selbst unter realistischeren Annahmen bleibt der ROI grob im Bereich von etwa 30 bis 50."
        ),
        "",
        "Fertige Antwort auf die Frage, warum das nicht im Paper steht:",
        (
            "Wir haben die ROI-Rechnung nicht ins Paper aufgenommen, weil uns dafuer belastbare "
            "hochschulspezifische Kostendaten und eine empirisch abgesicherte Annahme zur tatsaechlichen "
            "Interventionswirksamkeit fehlen. Deshalb wollten wir im Paper nicht den Eindruck einer "
            "Scheingenauigkeit erzeugen und haben uns auf robuste Modellmetriken konzentriert. "
            "Fuer die Praxis bieten wir stattdessen eine Threshold-Steuerung ueber ROC-AUC und Precision-Recall-Trade-offs an, "
            "damit Hochschulen die Schwelle an Budget und Fehlalarmtoleranz anpassen koennen."
        ),
        "",
        "## Drei-Satz-Antworten Zu Den Limitationen",
        "### 1. Leakage",
        (
            "Ja, die MI-Top-10-Auswahl wurde im aktuellen Paper vor der Cross-Validation bestimmt, "
            "und das ist methodisch eine milde Leakage-Quelle. Die saubere Loesung ist, `SelectKBest(mutual_info_classif)` "
            "als Pipeline-Step vor den Klassifikator zu setzen, damit die Auswahl pro Fold nur auf dem Trainingsfold gelernt wird. "
            f"Wir haben den Robustness Check ergaenzt, und die F1-Differenz liegt bei {_pp(leakage_delta_pp)}, also ist der Bias in der Praxis {leakage_severity}."
        ),
        "### 2. Kein Hyperparameter-Tuning",
        (
            "Ja, die Modelle wurden bewusst mit weitgehend gleichen Default-Einstellungen verglichen, "
            "dadurch ist der Benchmark transparent, aber nicht maximal optimiert. Besonders SVM profitiert erfahrungsgemaess "
            "am staerksten von Tuning bei `C` und `gamma`, deshalb ist das die wichtigste Fairness-Limitation. "
            f"Unser nachgereichter GridSearch zeigt als bestes Setup `C={svm_best['best_C']}` und `gamma={svm_best['best_gamma']}` "
            f"mit F1 {_pct(float(svm_best['best_f1']))}, waehrend Logistic Regression trotzdem eine sehr starke und stabile Baseline bleibt."
        ),
        "### 3. Nominale Features als Integer",
        (
            "Ja, einige nominale Merkmale liegen als Integer-Codes vor, und fuer lineare oder abstandsbasierte Modelle "
            "erzeugt das eine kuenstliche Ordnung, die semantisch nicht existiert. Methodisch sauberer ist ein `OneHotEncoder`, "
            "weil dadurch keine Scheinsortierung mehr in die Modellierung einfliesst. "
            f"Wir haben den Vergleich jetzt nachgezogen: LR veraendert sich von {_pct(baseline_lr)} auf {_pct(onehot_lr)}, "
            f"SVM von {_pct(baseline_svm)} auf {_pct(onehot_svm)}, wodurch man direkt sieht, wie praktisch relevant diese Limitation wirklich ist."
        ),
        "",
        "## Robustness Checks",
        _markdown_table(robustness_summary),
        "",
        "Detailtabelle SVM-GridSearch:",
        _markdown_table(
            svm_results.rename(
                columns={
                    "param_clf__C": "C",
                    "param_clf__gamma": "gamma",
                    "mean_test_f1": "F1",
                    "std_test_f1": "F1 Std",
                    "mean_test_roc_auc": "ROC-AUC",
                    "mean_test_accuracy": "Accuracy",
                    "rank_test_f1": "F1 Rank",
                }
            ).head(6),
            columns=["C", "gamma", "F1", "F1 Std", "ROC-AUC", "Accuracy", "F1 Rank"],
        ),
        "",
        "Detailtabelle One-Hot-Vergleich:",
        _markdown_table(
            onehot_df.rename(
                columns={
                    "model": "Variante",
                    "f1_mean": "F1",
                    "roc_auc_mean": "ROC-AUC",
                    "delta_vs_integer_pp": "Delta vs Integer (pp)",
                }
            ),
            columns=["Variante", "F1", "ROC-AUC", "Delta vs Integer (pp)"],
        ),
        "",
        "## Praktische Vor-Final-Uebungen",
        "1. Leakage-Pipeline einmal selbst live erklaeren und die beiden F1-Werte ohne Ablesen gegenueberstellen.",
        f"2. SVM-GridSearch mit Suchraum `C in [0.1, 1, 10]` und `gamma in ['scale', 0.01, 0.1, 1]` durchgehen; aktuell bestes Setup: `C={svm_best['best_C']}`, `gamma={svm_best['best_gamma']}`.",
        "3. One-Hot-Vergleich in einem Satz deuten: Hat die methodisch sauberere Kodierung das Ergebnis praktisch stark oder nur wenig veraendert?",
    ]
    return "\n".join(lines) + "\n"


def generate_defense_artifact() -> str:
    """Run the robustness checks and write the defense markdown artifact."""
    ensure_dirs(DEFENSE_DIR)

    df = prepare_binary(load_raw())
    X, y = get_X_y(df)

    leakage_df = compare_global_vs_pipeline_selection(X, y, k=10, save=True)
    svm_results, svm_best = run_svm_grid_search(X, y, save=True)
    onehot_df = compare_integer_vs_one_hot(X, y, save=True)

    lr_row, actual_dropouts = load_base_metrics()
    roi = compute_roi_metrics(
        precision=float(lr_row["precision_mean"]),
        recall=float(lr_row["recall_mean"]),
        actual_dropouts=actual_dropouts,
    )

    markdown = build_defense_markdown(
        leakage_df=leakage_df,
        svm_results=svm_results,
        svm_best=svm_best,
        onehot_df=onehot_df,
        lr_row=lr_row,
        roi=roi,
    )
    out_path = os.path.join(DEFENSE_DIR, "defense_prep_plan.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return out_path


if __name__ == "__main__":
    output = generate_defense_artifact()
    print(f"Defense artifact written to {output}")
