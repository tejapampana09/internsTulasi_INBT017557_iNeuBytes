# Task 2: Sentiment Analysis using Classical ML and DL

This project contains modular implementations of classical classifiers (Logistic Regression, SVM) and deep learning models (LSTMs) for binary sentiment analysis on the IMDb Movie Reviews dataset.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   python src/run_all_experiments.py
   ```
   This will:
   * Load the IMDb dataset from Hugging Face.
   * Split and clean text (seed 42).
   * Train Logistic Regression and SVM baselines.
   * Run 7 controlled studies (TF-IDF variants, custom LSTM embedding, GloVe pre-trained embeddings, dropout, and capacity).
   * Output comparative Markdown tables in `results/`.
   * Generate learning curves and confusion matrices in `plots/`.
   * Output the final mini-research paper report `REPORT.md`.

3. Generate diagrams:
   ```bash
   python src/generate_diagrams.py
   ```

## Directory Structure
* `src/data.py`: Preprocessing, cleaning, and partitioning.
* `src/models.py`: Classical and deep model factories.
* `src/train.py`: Evaluation metrics, plotting, error analysis, and GloVe loading.
* `src/run_all_experiments.py`: Master orchestrator.
* `src/generate_diagrams.py`: Architecture flowcharts.
