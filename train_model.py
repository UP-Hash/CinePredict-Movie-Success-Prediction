from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "tmdb_5000_movies.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
FEATURES = ["log_budget", "runtime", "release_year", "release_month", "main_genre", "original_language"]


def genre_name(value):
    try:
        parsed = json.loads(value) if isinstance(value, str) else []
        return parsed[0].get("name", "Unknown") if parsed else "Unknown"
    except Exception:
        return "Unknown"


def prepare_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA}. Run: python data/download_dataset.py")
    df = pd.read_csv(DATA)
    required = {"budget", "revenue", "genres", "runtime", "release_date", "original_language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    # Create the approved ROI label. Exclude invalid budget/revenue rows.
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df[(df["budget"] > 0) & (df["revenue"] >= 0)].copy()
    df["success"] = (df["revenue"] >= 2 * df["budget"]).astype(int)

    dates = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = dates.dt.year
    df["release_month"] = dates.dt.month
    df["main_genre"] = df["genres"].apply(genre_name)
    df["original_language"] = df["original_language"].fillna("unknown").astype(str)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["log_budget"] = np.log1p(df["budget"])
    df = df.dropna(subset=["release_year", "release_month"])
    df = df[(df["runtime"].isna()) | (df["runtime"] > 0)]
    return df[FEATURES], df["success"], df


def build_pipeline(model):
    numeric = ["log_budget", "runtime", "release_year", "release_month"]
    categorical = ["main_genre", "original_language"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])


def train_all(force=False):
    model_path = MODEL_DIR / "cinepredict_best.joblib"
    if model_path.exists() and not force:
        return joblib.load(model_path)

    X, y, raw = prepare_data()
    if y.nunique() < 2:
        raise ValueError("Dataset target has only one class after cleaning; cannot train a classifier.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            {"model__C": [0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [2, 5]}),
        "SVM": (SVC(class_weight="balanced", probability=True, random_state=42),
            {"model__C": [0.5, 1, 5], "model__kernel": ["rbf"]}),
        "KNN": (KNeighborsClassifier(),
            {"model__n_neighbors": [5, 9, 15]}),
    }
    rows, fitted = [], {}
    for name, (estimator, grid) in candidates.items():
        pipe = build_pipeline(estimator)
        search = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1, error_score="raise")
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "Best Parameters": str(search.best_params_),
        }
        rows.append(metrics)
        fitted[name] = search.best_estimator_

    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    best_name = results.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    bundle = {"model": best_model, "model_name": best_name, "features": FEATURES,
              "class_names": {0: "Flop", 1: "Hit"}, "threshold_rule": "Hit if revenue >= 2 * budget"}
    joblib.dump(bundle, model_path)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Best model: {best_name}\n\n" + classification_report(y_test, best_pred, target_names=["Flop", "Hit"], zero_division=0)
        + "\nConfusion matrix (rows=true, columns=predicted; Flop, Hit):\n" + str(confusion_matrix(y_test, best_pred)),
        encoding="utf-8")
    (REPORT_DIR / "dataset_summary.txt").write_text(
        f"Rows after cleaning: {len(X)}\nHit: {(y == 1).sum()}\nFlop: {(y == 0).sum()}\n"
        f"Excluded rows (invalid budget/revenue/date): {len(pd.read_csv(DATA)) - len(X)}\n",
        encoding="utf-8")
    return bundle

if __name__ == "__main__":
    bundle = train_all(force=True)
    print(f"Training complete. Best model: {bundle['model_name']}")
    print(f"Metrics: {REPORT_DIR / 'model_comparison.csv'}")
