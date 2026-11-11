# 💬 Sentiment Studio: NLP Sentiment Classification on Customer Reviews (TF-IDF)

A complete machine-learning project and Streamlit dashboard that classifies customer reviews as
**Positive** or **Negative**.

**Pipeline:** raw text → clean (HTML, lowercase, punctuation, stopwords, lemmatisation) → TF-IDF features →
**Multinomial Naive Bayes** and **Logistic Regression** → evaluation → saved models → live predictions.

## Project structure

```
sentiment_dashboard/
├── app.py              # Streamlit dashboard (6 pages)
├── train_model.py      # Trains + evaluates both models, saves them (CLI + function)
├── preprocessing.py    # Text cleaning (NLTK, with built-in fallback)
├── predict.py          # Loads saved models; predict_sentiment(), predict_batch()
├── config.py           # Shared paths and settings
├── requirements.txt
├── .streamlit/config.toml   # Dark theme
├── data/
│   ├── sentiment_reviews.csv      # file supplied with the task (10 rows, 1 neutral)
│   └── demo_sample_reviews.csv    # hand-written DEMO sample (64 rows)
└── models/             # trained model files are saved here
```

## Setup

Requires Python 3.9+.

```bash
cd sentiment_dashboard
python -m venv venv
# Windows: venv\Scripts\activate        macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet      # one-time download (the app also tries this itself)
```

## Run

```bash
python train_model.py        # 1. train once; saves models/ and metrics.json
streamlit run app.py         # 2. start the dashboard at http://localhost:8501
```

If you skip step 1, the dashboard shows a **Train** button on first launch. Models are trained only when
you ask, never on page refresh. Everything is loaded from `models/`.

## ⚠️ About the dataset (please read)

The task's file `sentiment_reviews.csv` has only **10 rows** (5 positive, 4 negative, 1 neutral), which is far
too small to train a reliable model. To let the project run out of the box, `data/demo_sample_reviews.csv`
adds 64 short, hand-written, balanced reviews. It is a **demo sample, not a benchmark**. By default both files
are merged (neutral rows are ignored, since this is a binary task), giving ≈73 reviews.

All metrics shown in the dashboard are computed from the trained models on a held-out test split. With
~73 reviews the test set has only ~15 reviews, so scores are noisy and the dashboard says so. It is expected
for them to be modest.

### Get reliable accuracy with a real dataset

1. Download the **IMDB Dataset of 50K Movie Reviews** from Kaggle (columns `review`, `sentiment`).
2. Save it as `data/IMDB_Dataset.csv`.
3. Train:
   ```bash
   python train_model.py --data data/IMDB_Dataset.csv
   # faster for a first try:
   python train_model.py --data data/IMDB_Dataset.csv --max-rows 10000
   ```
   Or use the **Model Performance → Train / retrain** panel in the dashboard to upload and train.

Any CSV works if it has a review-text column and a sentiment column with values like
`positive/negative` or `1/0`. Columns are auto-detected; override with `--text-col` and `--label-col`.

## Dashboard pages

| Page | What it does |
|---|---|
| 🏠 Home | Totals, class counts, best-model accuracy, distribution chart, statistics, recent predictions |
| 💬 Sentiment Analysis | Predict one review; probabilities, confidence gauge, example buttons, **Analyze** shows the words that drove the decision |
| 📦 Batch Analysis | Upload CSV → pick review column → predict all → table, charts, CSV download |
| 🔍 Dataset Explorer | Preview, missing values, class balance, review-length and word-frequency charts |
| 📊 Model Performance | Accuracy, precision, recall, F1, confusion matrices, classification reports, NB vs LR comparison, train vs test, retraining |
| 📘 About | Problem, objectives, methodology, preprocessing, TF-IDF, algorithms, workflow |

## Use the model from Python

```python
from predict import predict_sentiment
predict_sentiment("The battery died after two days, terrible.")
# {'label': 'negative', 'confidence': 0.6, 'probabilities': {...}, ...}
```
Or from the terminal: `python predict.py "I loved it" logistic_regression`

## Design decisions

- **No data leakage:** TF-IDF is fitted on the training split only.
- **Negations kept:** "not", "no", "never" are not removed as stopwords, so "not good" stays negative.
- **Macro-averaged** precision/recall/F1 (both classes weighted equally); per-class values are in the classification reports.
- **Preprocessing backend recorded:** the models remember whether NLTK or the built-in fallback was used. The app warns if that differs at prediction time.

## Troubleshooting

| Problem | Fix |
|---|---|
| NLTK download fails (no internet) | The app falls back to built-in stopwords/lemmatisation and shows a notice. Run `python -m nltk.downloader stopwords wordnet` later and retrain. |
| "Trained model files not found" | Run `python train_model.py` or use the Train button. |
| Models won't load after upgrading scikit-learn | Retrain with `python train_model.py`. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Training on a huge file is slow | Use `--max-rows 10000`. |
