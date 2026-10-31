# CinePredict — Movie Success Classification

A Streamlit ML mini-project that predicts **Hit / Flop** from pre-release movie metadata.

## Dataset used
**TMDB 5000 Movie Dataset** (Kaggle): https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

The project uses `tmdb_5000_movies.csv` (4,803 movie rows). The app can download the CSV automatically on first run from a public mirror. If that fails, download the file from Kaggle and place it at `data/tmdb_5000_movies.csv`.

## Hit / Flop definition
- Hit: `revenue >= 2 * budget`
- Flop: `revenue < 2 * budget`
- Rows with zero/missing budget, missing/invalid revenue, or invalid release date are excluded.

Revenue is used **only to create the target label**, never as an input feature. Popularity and vote fields are also excluded because they are post-release information. Features are budget, runtime, release year/month, main genre, and original language.

## Algorithms
1. Logistic Regression
2. Decision Tree
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)

Each model is tuned with 3-fold GridSearchCV. A stratified 80:20 train/test split is used. Best model is selected using F1-score. **Random Forest is not used.**

## Run locally (Windows)
1. Extract the ZIP.
2. Open the extracted `CinePredict` folder in VS Code.
3. Open terminal in that folder.
4. Create and activate virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Start the app:

```bash
streamlit run app.py
```

The browser will open at `http://localhost:8501`. First run downloads the dataset (internet required) and trains four models. Later launches reuse the saved best model.

## Run locally (Mac/Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Generated files
- `models/cinepredict_best.joblib` — selected trained model
- `reports/model_comparison.csv` — test metrics for all four models
- `reports/classification_report.txt` — classification report and confusion matrix
- `reports/dataset_summary.txt` — cleaned dataset class counts

## Notes
- First run needs internet if the dataset CSV is not already in `data/`.
- Dataset mirror: https://huggingface.co/datasets/adeel10/tmdb-movies
- Original dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- Educational project only; predictions are not guaranteed box-office outcomes.
