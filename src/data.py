"""Load the public UCI German Credit dataset."""

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_credit_data():
    dataset = fetch_ucirepo(id=144)
    X = dataset.data.features.copy()
    y = dataset.data.targets.squeeze().copy()
    if len(X) != len(y):
        raise ValueError("Features and labels have different lengths")
    if y.isna().any():
        raise ValueError("Target contains missing values")
    # UCI German Credit uses 1=good and 2=bad.
    values = set(pd.Series(y).unique())
    if values != {1, 2}:
        raise ValueError(f"Unexpected target labels: {values}")
    y = (y.astype(int) == 2).astype(int)
    y.name = "default"
    return X, y
