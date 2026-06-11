# Studienabbruch-Vorhersage (Student Dropout Prediction)

Data-Analytics-Projekt an der Hochschule Aalen, Studiengang Wirtschaftsinformatik, Sommersemester 2026.

> *Which Demographic, Socioeconomic, and Academic Factors Most Strongly Predict Student Dropout, and Which Classification Algorithm Achieves the Best Predictive Performance?*

## Team

- Maxim Werner
- Alexandra Pieri
- Sebastian Woecht
- Melih Zenen

## Worum geht es?

Wir untersuchen, welche Faktoren einen Studienabbruch am stärksten vorhersagen und welcher Klassifikationsalgorithmus dabei am besten abschneidet. Grundlage ist der öffentliche Datensatz *Predicting Student Dropout and Academic Success* (Realinho et al., 2022) einer portugiesischen Hochschule mit 4.424 Studierenden. Das Projekt folgt dem CRISP-DM-Prozessmodell und beantwortet zwei Forschungsfragen:

1. **RQ1:** Welche demografischen, sozioökonomischen und akademischen Faktoren sagen Studienabbruch am stärksten vorher?
2. **RQ2:** Welcher Klassifikationsalgorithmus erzielt die beste Vorhersageleistung?

Unsere Hypothese: Akademische Merkmale haben die größte Vorhersagekraft, gefolgt von sozioökonomischen und dann demografischen Merkmalen.

## Kernergebnisse

### RQ1: Was sagt Abbruch vorher?

Wir haben drei Feature-Selection-Methoden mit unterschiedlichen Annahmen auf alle 36 Eingabemerkmale angewendet: Mutual Information, ANOVA F-Score und Random Forest Feature Importance. Alle drei kommen zum selben Schluss: Die akademische Leistung im ersten Studienjahr dominiert die Vorhersage klar und bestätigt damit unsere Hypothese. Sechs Merkmale landen bei allen drei Methoden in den Top-Rängen:

| Rang | Merkmal | Gruppe |
|------|---------|--------|
| 1 | Curricular units 2nd sem (approved) | Academic |
| 2 | Curricular units 1st sem (approved) | Academic |
| 3 | Curricular units 2nd sem (grade) | Academic |
| 4 | Curricular units 1st sem (grade) | Academic |
| 5 | Tuition fees up to date | Socioeconomic |
| 6 | Age at enrollment | Demographic |

Makroökonomische Merkmale wie Arbeitslosenquote, Inflation und BIP tragen praktisch nichts zur Vorhersage bei.

### RQ2: Welches Modell gewinnt?

Logistic Regression schlägt unter 10-facher stratifizierter Kreuzvalidierung alle fünf Konkurrenten, darunter auch Random Forest und SVM:

| Metrik | Logistic Regression (alle 36 Features) |
|--------|------------------------------------------|
| Accuracy | 91,1 % ± 1,2 % |
| Precision | 93,0 % ± 1,9 % |
| Recall | 83,6 % ± 3,0 % |
| **F1-Score** | **88,0 % ± 1,8 %** |
| ROC-AUC | 95,3 % ± 0,4 % |

Dass ein lineares Modell die flexibleren Verfahren übertrifft, spricht dafür, dass die Entscheidungsgrenze zwischen Dropout und Graduate in diesem Merkmalsraum annähernd linear verläuft. Die Reduktion auf die Top-10-Features verbessert nur kNN (+5,1 % F1, Curse of Dimensionality) und minimal Naive Bayes. Alle anderen Modelle schneiden mit dem vollen Feature-Set besser ab.

## Datensatz

Quelle ist das UCI Machine Learning Repository: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697). Die Rohdaten liegen semikolongetrennt in [data/data.csv](data/data.csv), eine Beschreibung der Spalten in [data/data.txt](data/data.txt).

Der Originaldatensatz umfasst 4.424 Beobachtungen mit 36 Eingabemerkmalen und einer Zielvariable mit drei Klassen (Graduate, Dropout, Enrolled). Da wir ein binäres Vorhersageproblem betrachten, entfernen wir die Klasse Enrolled, weil diese Studierenden noch kein finales Ergebnis haben. Übrig bleiben 3.630 Beobachtungen: 1.421 Dropout (39,1 %, kodiert als 1) und 2.209 Graduate (60,9 %, kodiert als 0). Fehlende Werte gibt es keine.

Die 36 Merkmale sind in fünf Gruppen eingeteilt:

| Gruppe | Anzahl | Beispiele |
|--------|--------|-----------|
| Academic | 15 | Noten und bestandene Kurseinheiten (1. und 2. Semester), Zulassungsnote |
| Socioeconomic | 9 | Debtor, Tuition fees up to date, Scholarship, Bildung und Beruf der Eltern |
| Demographic | 8 | Alter bei Einschreibung, Geschlecht, Nationalität, Application mode |
| Macroeconomic | 3 | Arbeitslosenquote, Inflation, BIP |
| Other | 1 | Course (Studiengang) |

## Projektstruktur

```
Code/
├── data/
│   ├── data.csv                  # Rohdatensatz (semikolongetrennt)
│   └── data.txt                  # Beschreibung der Spalten
├── notebooks/                    # Auswertung in 4 Schritten, in dieser Reihenfolge ausführen
│   ├── 01_exploration.ipynb      # EDA: Klassenverteilung, Verteilungen, Korrelationen
│   ├── 02_preprocessing_and_feature_selection.ipynb  # Vorverarbeitung, MI / F-Score / RFFI
│   ├── 03_modeling.ipynb         # 10-Fold-CV für alle 6 Modelle (36 Features und Top-10)
│   └── 04_results_and_interpretation.ipynb           # Evaluation, Publikationsgrafiken
├── src/                          # Wiederverwendbare Projektmodule
│   ├── config.py                 # Zentrale Konfiguration: Pfade, Konstanten, Featuregruppen
│   ├── data_loader.py            # Laden, Filtern (Enrolled raus), binäre Kodierung
│   ├── preprocessing.py          # sklearn-Pipelines (Imputation, optional StandardScaler)
│   ├── feature_selection.py      # Mutual Information, ANOVA F-Score, RF-Importance
│   ├── train_models.py           # Modelldefinitionen, CV-Runner, Robustheitschecks
│   ├── evaluate_models.py        # Post-hoc-Evaluation, PR-Kurve, LR-Koeffizienten
│   ├── defense_prep.py           # Robustheitschecks für die Verteidigung (siehe unten)
│   ├── plots.py                  # Alle Visualisierungsfunktionen
│   └── utils.py                  # Hilfsfunktionen (Verzeichnisse, Speichern)
├── results/
│   ├── exploration/              # 13 EDA-Grafiken
│   ├── figures/                  # Publikationsgrafiken (Konfusionsmatrix, ROC, Rankings, ...)
│   ├── tables/                   # CSV- und JSON-Ergebnistabellen
│   ├── models/
│   │   └── best_model.pkl        # Bestes Modell (Logistic Regression, serialisiert)
│   └── defense/                  # Robustheitschecks zur Verteidigung
├── requirements.txt
└── README.md
```

## Methodik

### Vorverarbeitung

Nach dem Entfernen der Klasse Enrolled wird die Zielvariable binär kodiert (Dropout = 1, Graduate = 0). Alle Pipelines enthalten eine Median-Imputation, obwohl der Datensatz keine fehlenden Werte hat. Das macht den Workflow robust gegenüber zukünftigen Daten. Die z-Standardisierung wird nur bei skalenabhängigen Modellen angewendet (Logistic Regression, SVM, kNN). Entscheidungsbäume, Random Forest und Naive Bayes arbeiten auf den unskalierten Daten, weil ihre Vorhersagen von monotonen Transformationen nicht beeinflusst werden.

Alle Schritte stecken in sklearn-Pipelines: Imputer und Scaler werden nur auf den Trainings-Folds gefittet und auf den Test-Fold lediglich angewendet. So kann keine Information aus den Testdaten in die Vorverarbeitung gelangen (kein Data Leakage).

### Feature-Selektion

Weil jede Methode andere Annahmen trifft, vergleichen wir drei komplementäre Verfahren. Verglichen werden dabei nur die Rangfolgen, nicht die Score-Werte, da diese auf völlig unterschiedlichen Skalen liegen:

1. **Mutual Information (MI):** informationstheoretischer Filter, erfasst lineare und nichtlineare Zusammenhänge
2. **ANOVA F-Score:** univariater statistischer Filter, vergleicht die Klassenmittelwerte
3. **Random Forest Feature Importance (RFFI):** eingebettete Methode auf Basis der Gini-Impurity-Reduktion

Aus dem MI-Ranking wird ein Top-10-Subset gebildet (Dimensionsreduktion um rund 72 %) und gegen das volle Feature-Set getestet. Ein Konsens-Ranking, das die Rangpositionen über alle drei Methoden mittelt, identifiziert die stabilsten Prädiktoren.

### Klassifikationsmodelle

| Modell | Typ | Skaliert | Einstellungen |
|--------|-----|----------|---------------|
| Logistic Regression | Linear | Ja | C=1.0, max_iter=1000 |
| Decision Tree | Baum | Nein | max_depth=None |
| Random Forest | Ensemble | Nein | n_estimators=100 |
| Gaussian Naive Bayes | Probabilistisch | Nein | Gaussian Likelihoods |
| SVM | Kernel | Ja | C=1.0, kernel=rbf, gamma=scale |
| kNN | Distanzbasiert | Ja | n_neighbors=5 |

Auf eine Hyperparameter-Suche wurde im Hauptexperiment bewusst verzichtet, alle Modelle laufen mit den oben gelisteten Einstellungen (siehe Limitationen).

### Evaluation

Alle Modelle werden mit 10-facher stratifizierter Kreuzvalidierung bewertet (`StratifiedKFold`, `random_state=42`). Die Stratifizierung erhält das Klassenverhältnis von etwa 61:39 in jedem Fold, und jede Beobachtung landet genau einmal im Test-Fold. Berechnet werden Accuracy, Precision, Recall, F1 und ROC-AUC, jeweils als Mittelwert mit Standardabweichung über die Folds und bezogen auf die positive Klasse Dropout.

Der F1-Score ist unser primäres Auswahlkriterium, weil er Recall (gefährdete Studierende erkennen) und Precision (Fehlalarme vermeiden) ausbalanciert. Die Feature-Selektion ist über `SelectKBest` in jede CV-Fold eingebettet, sodass die Testdaten die Selektion nicht beeinflussen können.

### Voller Feature-Satz gegen Top-10 (F1)

| Modell | Alle 36 | Top-10 | ΔF1 |
|--------|---------|--------|------|
| Logistic Regression | **88,0 %** | 85,4 % | −2,6 % |
| Random Forest | 87,5 % | 85,7 % | −1,8 % |
| SVM (RBF) | 86,7 % | 85,7 % | −1,0 % |
| Decision Tree | 82,0 % | 80,2 % | −1,8 % |
| kNN | 78,5 % | **83,6 %** | **+5,1 %** |
| Naive Bayes | 78,2 % | 78,7 % | +0,5 % |

## Reproduzierbarkeit

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### Ausführung

Die vier Notebooks in dieser Reihenfolge ausführen:

1. [notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb)
2. [notebooks/02_preprocessing_and_feature_selection.ipynb](notebooks/02_preprocessing_and_feature_selection.ipynb)
3. [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb)
4. [notebooks/04_results_and_interpretation.ipynb](notebooks/04_results_and_interpretation.ipynb)

Alle Zufallskomponenten verwenden `random_state=42`, die Ergebnisse sind damit vollständig reproduzierbar. Erzeugte Tabellen und Grafiken landen automatisch in `results/`.

### Abhängigkeiten

Siehe [requirements.txt](requirements.txt): pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, jupyter, notebook, ipykernel und joblib.

## Erzeugte Ergebnisse

### EDA-Grafiken ([results/exploration/](results/exploration/))

13 Grafiken zur Datenexploration: Klassenverteilung, Alters- und Notenverteilungen nach Zielklasse, sozioökonomische Merkmale, Korrelationsmatrix, Pairplot, Abbruchraten je Kategorie und die Korrelationsheatmap der Top-10-Features.

### Publikationsgrafiken ([results/figures/](results/figures/))

| Datei | Inhalt |
|-------|--------|
| `confusion_matrix_best_model.png` | Konfusionsmatrix (aggregiert über 10 Folds) |
| `roc_curve_best_model.png` | ROC-Kurve mit AUC-Annotation |
| `precision_recall_best_model.png` | Precision-Recall-Kurve (AP = 0,9499) |
| `model_f1_comparison.png` / `model_accuracy_comparison.png` | Modellvergleiche |
| `radar_model_comparison.png` | Radar-Chart: alle 6 Modelle, 4 Metriken |
| `cv_boxplot_f1.png` / `cv_boxplot_roc.png` | CV-Stabilität je Modell |
| `all_vs_top10_comparison.png` | Alle 36 gegen Top-10 Features |
| `feature_importance_top10_mi.png` / `feature_importance_top10.png` | Top-10-Rankings (MI bzw. RF) |
| `feature_importance_consensus.png` | Konsens-Heatmap MI / F-Score / RF |
| `group_importance.png` | Mittlere Wichtigkeit je Featuregruppe |
| `lr_coefficients.png` | Standardisierte LR-Koeffizienten |
| `per_class_metrics.png` | Precision, Recall, F1 je Klasse |
| `correlation_heatmap_clean.pdf` | Bereinigte Korrelationsheatmap (Vektorversion für das Paper, Fig. 1) |

### Tabellen ([results/tables/](results/tables/))

CV-Ergebnisse für alle Features und das Top-10-Subset (jeweils als CSV-Zusammenfassung und Per-Fold-JSON), die Feature-Importance-Scores aller drei Methoden, eine gruppierte Feature-Zusammenfassung, die Top-Feature-Subsets und deskriptive Statistiken des Datensatzes.

### Robustheitschecks ([results/defense/](results/defense/))

Über [src/defense_prep.py](src/defense_prep.py) erzeugte Zusatzanalysen, die die Hauptergebnisse für die Projektverteidigung absichern:

- `onehot_vs_integer_comparison.csv` vergleicht One-Hot- und Integer-Kodierung der nominalen Merkmale
- `global_top10_vs_pipeline_selector.csv` vergleicht globale und fold-interne Feature-Selektion (Data-Leakage-Check)
- `svm_grid_search_results.csv` und `svm_grid_search_best_params.csv` dokumentieren die SVM-Hyperparameter-Suche

## Interpretation und Limitationen

Der Studienfortschritt im ersten Studienjahr, also bestandene Kurseinheiten und Noten, ist der primäre Treiber des Abbruchrisikos. Das passt zu Tintos Modell der studentischen Integration, nach dem akademische Integration der wichtigste Bestimmungsfaktor für Abbruch ist. Sozioökonomische Faktoren wie Tuition fees, Scholarship und Debtor liefern ein verlässliches Sekundärsignal finanzieller Prekarität. Demografische Merkmale folgen auf Platz drei, wobei nur das Alter bei Einschreibung über alle Methoden hinweg stabil ist.

Praktisch bedeutet ein Recall von 83,6 %, dass das Modell etwa fünf von sechs späteren Abbrecher:innen nach dem ersten Studienjahr korrekt erkennt. Das ist eine brauchbare Basis für ein Frühwarnsystem mit gezielten Interventionen wie Beratung, finanzieller Unterstützung oder Mentoring.

Im Paper diskutierte Limitationen:

- Keine Hyperparameter-Optimierung im Hauptexperiment (die SVM ist vermutlich durch die Default-Werte für γ und C limitiert)
- Das Klassenungleichgewicht von 61:39 wurde nicht explizit behandelt, etwa über `class_weight='balanced'`
- Nominale Merkmale wie Course und Application mode sind als Integer-Codes statt One-Hot kodiert
- Für eine vollständig unverzerrte Generalisierungsschätzung wäre eine äußere CV-Schleife nötig

## Begleitendes Paper

Die vollständige wissenschaftliche Ausarbeitung im IEEE-Format trägt den Titel *Which Demographic, Socioeconomic, and Academic Factors Most Strongly Predict Student Dropout, and Which Classification Algorithm Achieves the Best Predictive Performance?* von Maxim Werner, Alexandra Pieri, Sebastian Woecht und Melih Zenen (Hochschule Aalen, Dept. of Information Systems).
