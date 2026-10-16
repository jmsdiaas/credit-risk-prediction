import numpy as np
import pandas as pd
from src.train import evaluate, make_preprocessor


def test_evaluate_perfect_classifier():
    metrics = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_preprocessor_mixed_types():
    data = pd.DataFrame({"age": [20, 40, None], "job": ["A", "B", "A"]})
    transformed = make_preprocessor(data).fit_transform(data)
    assert transformed.shape[0] == 3
