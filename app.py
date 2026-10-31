from pathlib import Path
import sys
import subprocess

import streamlit as st
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_model import train_all, prepare_data, REPORT_DIR, DATA


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CinePredict | Movie Success AI",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background: #0b1020;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #151b32 100%
        );
    }

    h1, h2, h3, p, label {
        color: #f4f6ff;
    }

    div[data-testid='stMetric'] {
        background: #171f38;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #303b5c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎬 CinePredict")

st.subheader(
    "Movie Success Classification Using Machine Learning"
)

st.caption(
    "Predict movie success using pre-release movie details."
)


# --------------------------------------------------
# Load and Train Models
# --------------------------------------------------

@st.cache_resource
def load_bundle():
    return train_all()


# --------------------------------------------------
# Dataset Check
# --------------------------------------------------

if not DATA.exists():

    st.warning(
        "Dataset not found. Downloading the TMDB dataset "
        "(internet connection required)..."
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "data" / "download_dataset.py")
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        st.error(
            result.stdout + "\n" + result.stderr
        )

        st.info(
            "Manual option: Download "
            "tmdb_5000_movies.csv from Kaggle, "
            "place it in the CinePredict/data/ folder, "
            "and rerun the app."
        )

        st.stop()


# --------------------------------------------------
# Prepare Data and Models
# --------------------------------------------------

try:

    with st.spinner(
        "Training or loading machine learning models..."
    ):

        bundle = load_bundle()

        X, y, raw = prepare_data()

except Exception as exc:

    st.error(
        f"Project setup or training error: {exc}"
    )

    st.stop()


# --------------------------------------------------
# Model Comparison Data
# --------------------------------------------------

comparison_path = REPORT_DIR / "model_comparison.csv"

if comparison_path.exists():

    comparison = pd.read_csv(
        comparison_path
    )

else:

    comparison = pd.DataFrame()


# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------

m1, m2, m3 = st.columns(3)

m1.metric(
    "Movies Used",
    f"{len(X):,}"
)

m2.metric(
    "Hit Movies",
    f"{int((y == 1).sum()):,}"
)

m3.metric(
    "Best Performing Model",
    bundle["model_name"]
)


# --------------------------------------------------
# Navigation Tabs
# --------------------------------------------------

predict_tab, results_tab, about_tab = st.tabs(
    [
        "🔮 Predict Movie",
        "📊 Model Results",
        "ℹ️ About Project"
    ]
)


# ==================================================
# TAB 1: PREDICT MOVIE
# ==================================================

with predict_tab:

    st.markdown(
        "### Enter Movie Details"
    )

    st.write(
        "Predictions use pre-release features only. "
        "Revenue, popularity, and vote statistics "
        "are not used as input features."
    )

    genres = sorted(
        set(
            raw["main_genre"]
            .dropna()
            .astype(str)
        ) - {"Unknown"}
    )

    languages = sorted(
        set(
            raw["original_language"]
            .dropna()
            .astype(str)
        )
    )

    c1, c2 = st.columns(2)

    # ----------------------------------------------
    # Movie Information
    # ----------------------------------------------

    with c1:

        budget = st.number_input(
            "Estimated Production Budget (USD)",
            min_value=10000,
            max_value=500000000,
            value=30000000,
            step=1000000
        )

        runtime = st.number_input(
            "Runtime (Minutes)",
            min_value=30,
            max_value=300,
            value=120,
            step=5
        )

        year = st.number_input(
            "Release Year",
            min_value=1950,
            max_value=2035,
            value=2026,
            step=1
        )

    # ----------------------------------------------
    # Release Information
    # ----------------------------------------------

    with c2:

        month = st.selectbox(
            "Release Month",
            list(range(1, 13)),
            index=5,
            format_func=lambda x: pd.Timestamp(
                2000, x, 1
            ).strftime("%B")
        )

        genre = st.selectbox(
            "Main Genre",
            genres if genres else ["Drama"]
        )

        language = st.selectbox(
            "Original Language",
            languages if languages else ["en"],
            index=(
                languages.index("en")
                if "en" in languages
                else 0
            )
        )

    # ----------------------------------------------
    # Prediction Button
    # ----------------------------------------------

    if st.button(
        "Predict Movie Success",
        type="primary",
        use_container_width=True
    ):

        row = pd.DataFrame(
            [
                {
                    "log_budget": np.log1p(budget),
                    "runtime": runtime,
                    "release_year": year,
                    "release_month": month,
                    "main_genre": genre,
                    "original_language": language
                }
            ]
        )

        model = bundle["model"]

        prediction = int(
            model.predict(row)[0]
        )

        probabilities = model.predict_proba(
            row
        )[0]

        confidence = (
            float(probabilities[prediction]) * 100
        )

        # ------------------------------------------
        # Display Prediction
        # ------------------------------------------

        if prediction == 1:

            st.success(
                f"🎉 Prediction: HIT | "
                f"Model Confidence: {confidence:.1f}%"
            )

            st.write(
                "The model predicts that the movie "
                "may earn at least twice its "
                "production budget."
            )

        else:

            st.error(
                f"Prediction: FLOP | "
                f"Model Confidence: {confidence:.1f}%"
            )

            st.write(
                "The model predicts that the movie "
                "may earn less than twice its "
                "production budget."
            )

        st.caption(
            "Confidence represents the model's "
            "estimated probability. It is not "
            "a guaranteed outcome."
        )


# ==================================================
# TAB 2: MODEL RESULTS
# ==================================================

with results_tab:

    st.markdown(
        "### Model Comparison "
        "(20% Held-Out Test Set)"
    )

    if not comparison.empty:

        st.dataframe(
            comparison[
                [
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ]
            ].style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1 Score": "{:.3f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            comparison.set_index("Model")[
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ]
            ]
        )

    else:

        st.info(
            "Model comparison results are not available yet."
        )

    report = (
        REPORT_DIR / "classification_report.txt"
    )

    if report.exists():

        with st.expander(
            "Best Model Classification Report "
            "and Confusion Matrix"
        ):

            st.code(
                report.read_text(
                    encoding="utf-8"
                )
            )

    st.caption(
        "Model tuning uses 3-fold GridSearchCV. "
        "Random Forest is not used."
    )


# ==================================================
# TAB 3: ABOUT PROJECT
# ==================================================

with about_tab:

    st.markdown(
        "### Project Details"
    )

    st.markdown(
        "**Dataset:** TMDB 5000 Movie Dataset "
        "from Kaggle (`tmdb_5000_movies.csv`)."
    )

    st.markdown(
        "**Target Rule:** A movie is classified as "
        "Hit if revenue is at least twice its budget. "
        "Otherwise, it is classified as Flop."
    )

    st.markdown(
        "**Data Cleaning:** Rows with zero or missing "
        "budget, missing or invalid revenue, or "
        "invalid release dates are excluded."
    )

    st.markdown(
        "**Features:** Log budget, runtime, release "
        "year, release month, main genre, "
        "and original language."
    )

    st.markdown(
        "**Algorithms:** Logistic Regression, "
        "Decision Tree, Support Vector Machine (SVM), "
        "and K-Nearest Neighbors (KNN)."
    )

    st.markdown(
        "**Leakage Prevention:** Revenue is used "
        "only to create the target label. It is never "
        "passed to the model. Post-release popularity "
        "and vote statistics are also excluded."
    )

    st.markdown(
        "**Evaluation:** Stratified 80:20 train-test "
        "split, 3-fold hyperparameter tuning, "
        "Accuracy, Precision, Recall, F1-score, "
        "and Confusion Matrix."
    )

    st.info(
        "This is an educational classification model "
        "trained on historical TMDB data. "
        "It does not guarantee real-world "
        "box-office performance."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "CinePredict | B.Tech CSE Machine Learning Mini Project"
)