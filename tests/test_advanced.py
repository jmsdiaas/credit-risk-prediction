import numpy as np
from src.advanced import business_cost, find_best_threshold


def test_cost_confusion_matrix():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.7, 0.1, 0.8])
    r = business_cost(y, p, 0.5, 10000, 1500)
    assert (r['tn'], r['fp'], r['fn'], r['tp']) == (1, 1, 1, 1)
    assert r['cost'] == 11500


def test_threshold_optimized_only_on_supplied_labels():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.9])
    best, curve = find_best_threshold(y, p)
    assert best['cost'] == 0
    assert len(curve) == 99
