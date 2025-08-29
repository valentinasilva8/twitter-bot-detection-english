## Twitter Bot Detection (Human vs Bot) – English Dataset

### Overview
Detects automated (bot) accounts on Twitter using supervised machine learning (Logistic Regression, SVM, KNN, Random Forest), deep learning (hybrid and bidirectional LSTM combining text and account metadata), and unsupervised learning (PCA + KMeans). The project provides an end‑to‑end pipeline: data inspection, outlier analysis, preprocessing and scaling, feature selection, model training/evaluation, and clustering-based analysis.

### Project Motivation
As AI‑generated and automated accounts proliferate on social platforms, timely and reliable bot detection is essential for content integrity and user safety. This project explores a spectrum of classical ML, deep learning, and clustering approaches to compare modeling strategies and feature contributions for English‑language Twitter accounts.

### Datasets
- `twitter_english_data.csv`: Labeled numeric + metadata features for supervised models and clustering.
  - Label: `0 = Human`, `1 = Bot` (see `main.py`).
  - Example numeric/metadata features (from code usage): `num_links`, `num_hashtags`, `num_mentions`, `num_chars`, `num_followers`, `num_friends`, `is_verified`, `account_age`, `tweet_length`, `statuses_count`, `favorites_count`, `listed_count`, `friends_follower_ratio`.
- `english_data.csv`: Labeled dataset with `processed_tweet` (tokenized text) plus the numeric features above for LSTM models.
  - Split sizes (per report/code): ~70/15/15 for LSTM (train/dev/test), ~80/20 for classical ML.
  - Approximate scale (per report): ~5,000 training accounts; ~1,000 each for validation and test.

### Features & Preprocessing
- Data info and summary statistics: `Info.py`, `StatAnalysis.py`.
- Outlier reporting (IQR method), without removal due to prevalence: `OutlierAnalysis.py`.
- Scaling: `PreprocessData.py` applies `RobustScaler` then `MinMaxScaler` to numeric columns (excluding `label`, `is_verified`).
- Train/test split: consistent split via `SplittingDataset.py` (80/20).
- Exploratory analysis: label distribution, correlation heatmap, pairplots of top features in `ExploratoryAnalysis.py`.
- Feature selection for classical ML: correlation‑based top‑k subsets in `DifferentTrainingSets.py` (top‑5, top‑10, and full set).

### Models & Methods
- Supervised ML (trained on scaled numeric features):
  - `LogisticRegression.py` (scikit‑learn Logistic Regression)
  - `SVM.py` (linear SVC with probability=True)
  - `KNN.py` (KNeighborsClassifier)
  - `RandomForrest.py` (RandomForestClassifier with class_weight, SMOTEENN resampling, and RandomizedSearchCV hyperparameter tuning; decision threshold tuning at 0.6)
- Deep Learning (text + numeric features): `LSTM.py`
  - Hybrid LSTM: text branch (Embedding + stacked LSTM + dropout) concatenated with numeric branch (Dense + dropout); trained with ADASYN oversampling; callbacks include EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint → `best_hybrid_lstm_model.keras`.
  - Bidirectional LSTM: BiLSTM text branch + numeric features (including TextBlob sentiment subjectivity); ADASYN oversampling; EarlyStopping + ModelCheckpoint → `best_lstm_model.keras`.
- Unsupervised Learning: `Kmeans_pca.py`
  - PCA retains 80% and 90% variance, then KMeans (k=2) on PCA scores; results merged back with labels for comparison and plotted in 2D/3D.

### Evaluation
- Centralized evaluation in `EvaluateModel.py` (ROC‑AUC, classification report; confusion matrix plots for all models; top‑5 feature importances plot for Random Forest).
- For clustering, `main.py` computes classification reports and confusion matrices comparing KMeans clusters against ground truth labels after PCA.
- Note: Numerical scores are printed at runtime; this repository does not contain fixed metrics tables.

### Results Summary
- From the accompanying report (Twitter English dataset):
  - KNN accuracy: 0.86
  - Logistic Regression accuracy: 0.87
  - SVM (RBF) accuracy: 0.87
  - Random Forest accuracy: 0.86
  - Bi‑LSTM accuracy: 0.87
  - PCA + KMeans clustering accuracy: 0.88 (highest reported in the study)
- From `training.ipynb` runs:
  - Hybrid LSTM (text + numeric): test accuracy ≈ 0.857, ROC‑AUC ≈ 0.921
  - Bidirectional LSTM (text + numeric + sentiment): test accuracy ≈ 0.872, ROC‑AUC ≈ 0.911

Notes from the study:
- Deep learning approaches outperform classical ML across precision/recall/F1, while clustering provides complementary label‑free insights.
- `is_verified` surfaced as a strong predictor in tree models; however, its reliability is reduced under paid verification policies.

### Repository Entry Points
- `main.py` orchestrates the full pipeline on `data/twitter_english_data.csv` and runs LSTM experiments on `data/english_data.csv`:
  1) Inspect/describe data → 2) Outlier reporting → 3) Scaling → 4) Split → 5) EDA → 6) Build top‑k feature subsets → 7) Train Logistic Regression, SVM, KNN, Random Forest → 8) Evaluate → 9) Train and evaluate Hybrid/BiLSTM → 10) PCA + KMeans clustering and evaluation.

### Tech Stack
- Python, pandas, NumPy, seaborn, Matplotlib
- scikit‑learn (Logistic Regression, SVC, KNN, RandomForest, PCA, KMeans, train_test_split, metrics)
- imbalanced‑learn (ADASYN, SMOTEENN)
- TensorFlow/Keras (Embedding, LSTM, Bidirectional, Model, callbacks)
- NLP utilities: NLTK, TextBlob
- Plotly (3D PCA cluster visualization)

### Setup & How to Run
Prerequisites: Python 3.9+ recommended.

1) Create environment and install dependencies
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn imbalanced-learn tensorflow keras nltk seaborn matplotlib plotly textblob langdetect wordcloud
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

2) Prepare data files in the project root
- `data/twitter_english_data.csv` (for classical ML and clustering)
- `data/english_data.csv` (for LSTM experiments)

3) Run the full pipeline
```
python main.py
```
The script prints ROC‑AUC and classification reports, and renders confusion matrices and feature importance plots. LSTM checkpoints are saved as `best_hybrid_lstm_model.keras` and `best_lstm_model.keras`.

### Visuals
- EDA: label distribution bar chart, correlation heatmap, pairplots (top features).
- Model evaluation: confusion matrices; Random Forest top‑5 feature importance.
- Clustering: PCA variance curve, 2D scatter by cluster, 3D PCA cluster visualization (Plotly).

### Notes & Reproducibility
- Labels are binary with `0 = Human`, `1 = Bot`.
- Class imbalance is addressed with ADASYN (LSTM) and SMOTEENN (Random Forest).
- Random seeds used where applicable (e.g., `random_state=42`), but some components (GPU ops, multithreading) may introduce minor nondeterminism.

### License
No license specified in the repository.


