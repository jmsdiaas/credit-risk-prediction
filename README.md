# Credit Risk Prediction | Explainable Machine Learning

A reproducible, portfolio-ready baseline for credit default prediction using the public **UCI Statlog (German Credit Data)** dataset.

## Highlights
- Data download from the UCI Machine Learning Repository (1,000 observations, 20 predictors, binary credit outcome)
- Leakage-aware stratified train/test split
- Preprocessing pipelines for numeric and categorical features
- Logistic Regression and Random Forest baselines
- ROC-AUC, average precision, confusion matrix and classification report
- Configurable decision threshold and exported metrics
- Unit tests and GitHub Actions CI

> **Note:** This is a learning portfolio project, not a production credit decision system. It does not make fairness, regulatory compliance, or deployment claims.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
pytest -q
```

The training command downloads the dataset and writes `outputs/metrics.json`, `outputs/confusion_matrix.png`, and `outputs/model.joblib`. The `outputs/` directory is gitignored.

## Dataset

UCI Machine Learning Repository, *Statlog (German Credit Data)*: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The dataset is fetched via `ucimlrepo`; no data or trained artifacts are committed to the repository.

## Methodology

1. Fetch and validate the dataset.
2. Convert the outcome to `1 = bad credit risk` and `0 = good credit risk`.
3. Split data with stratification before fitting preprocessors.
4. Fit and compare Logistic Regression and Random Forest on the same held-out test set.
5. Report ranking metrics and threshold-dependent metrics separately.

**Caveats:** A small historic dataset; a single held-out test split gives noisy estimates. For serious applications add cross-validation, calibration, temporal validation, fairness analysis, and cost-sensitive decision thresholds. AUC scores are not guaranteed and depend on the run.

## Structure

```text
src/data.py        # dataset retrieval and validation
src/train.py       # training, evaluation, exports
tests/             # basic automated tests
.github/workflows/ # CI
```

## Next milestones

- Cross-validation and hyperparameter tuning
- Model interpretability (permutation importance and SHAP)
- Cost-sensitive threshold analysis
- Streamlit demonstration with synthetic sample inputs
- Model card and limitations

## License
MIT
