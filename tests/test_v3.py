import numpy as np
import pytest
from src.v3 import expected_calibration_error, bootstrap_intervals, score

def test_perfect_calibration():
    assert expected_calibration_error([0,1],[0,1]) == pytest.approx(0)

def test_ece_rejects_invalid_probabilities():
    with pytest.raises(ValueError): expected_calibration_error([0,1],[-.1,.5])

def test_cost_matrix_direction():
    result=score([0,0,1,1],[.1,.8,.2,.9],.5)
    assert result['fp']==1 and result['fn']==1 and result['cost']==11500

def test_bootstrap_deterministic():
    y=np.array([0,0,0,1,1,1]);p=np.array([.1,.2,.6,.4,.7,.9])
    a=bootstrap_intervals(y,p,.5,iterations=25,seed=123)
    assert a==bootstrap_intervals(y,p,.5,iterations=25,seed=123)
    assert a['roc_auc']['lower_95'] <= a['roc_auc']['upper_95']
