"""V3: probability calibration, validation-only selection, and test uncertainty.

Educational research only. Monetary assumptions are hypothetical.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.data import load_credit_data
from src.train import make_preprocessor
from src.advanced import business_cost, find_best_threshold

SEED = 42

def expected_calibration_error(y, probabilities, bins=10):
    """Equal-width-bin ECE; descriptive only, sensitive to bin count."""
    y = np.asarray(y, dtype=int); p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or len(y) == 0: raise ValueError("Invalid arrays")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)): raise ValueError("Invalid probabilities")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.sum(idx == k) / len(y) * abs(y[idx == k].mean() - p[idx == k].mean())
                     for k in range(bins) if np.any(idx == k)))

def score(y, p, threshold, default_cost=10000, rejection_cost=1500):
    result = business_cost(y, p, threshold, default_cost, rejection_cost)
    result.update(roc_auc=float(roc_auc_score(y, p)),
                  average_precision=float(average_precision_score(y, p)),
                  brier=float(brier_score_loss(y, p)),
                  log_loss=float(log_loss(y, p, labels=[0, 1])),
                  ece_10=float(expected_calibration_error(y, p)))
    result['cost_per_applicant'] = result['cost'] / len(y)
    return result

def bootstrap_intervals(y, p, threshold, iterations=1000, seed=SEED,
                        default_cost=10000, rejection_cost=1500):
    """Stratified paired resampling on locked test predictions; no reselection."""
    y=np.asarray(y, dtype=int); p=np.asarray(p, dtype=float)
    if len(np.unique(y)) != 2: raise ValueError("Both classes required")
    rng=np.random.default_rng(seed)
    classes=[np.flatnonzero(y == c) for c in (0, 1)]
    samples={k: [] for k in ('roc_auc', 'average_precision', 'brier', 'cost_per_applicant')}
    for _ in range(iterations):
        ix=np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        r=score(y[ix],p[ix],threshold,default_cost,rejection_cost)
        for k in samples: samples[k].append(r[k])
    return {k: {'lower_95':float(np.quantile(v,.025)), 'upper_95':float(np.quantile(v,.975))}
            for k,v in samples.items()}

def make_models(X):
    estimators={
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced',random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=250,min_samples_leaf=3,
                                                class_weight='balanced',random_state=SEED,n_jobs=-1)}
    return {name:Pipeline([('preprocessor',make_preprocessor(X)),('model',est)])
            for name,est in estimators.items()}

def main(output_dir='outputs_v3', default_cost=10000, rejection_cost=1500, iterations=1000):
    X,y=load_credit_data()
    # Identical held-out 20% test as V2. No test inspection during selection.
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    # 60% training/calibration pool, 20% validation, 20% test.
    X_fit,X_val,y_fit,y_val=train_test_split(X_dev,y_dev,test_size=.25,stratify=y_dev,random_state=SEED)
    # Calibration folds occur strictly inside the fit pool.
    candidates={}; validation={}
    for name,pipeline in make_models(X_fit).items():
        for method in ('sigmoid','isotonic'):
            label=f'{name}__{method}'
            model=CalibratedClassifierCV(estimator=pipeline,method=method,cv=3,ensemble=True)
            model.fit(X_fit,y_fit)
            p=model.predict_proba(X_val)[:,1]
            optimum,_=find_best_threshold(y_val,p,default_cost,rejection_cost)
            validation[label]=score(y_val,p,optimum['threshold'],default_cost,rejection_cost)
            candidates[label]=model
    # Predeclared selection: minimize validation cost; Brier and AUC break ties.
    winner=min(validation,key=lambda k:(validation[k]['cost'],validation[k]['brier'],-validation[k]['roc_auc'],k))
    threshold=validation[winner]['threshold']
    # Lock trained candidate and threshold before first test evaluation.
    model=candidates[winner]
    p_test=model.predict_proba(X_test)[:,1]
    result=score(y_test,p_test,threshold,default_cost,rejection_cost)
    baseline=score(y_test,p_test,.5,default_cost,rejection_cost)
    ci=bootstrap_intervals(y_test,p_test,threshold,iterations,SEED,default_cost,rejection_cost)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    payload={'dataset':'UCI Statlog German Credit (id=144)', 'seed':SEED,
      'sample_counts':{'fit_and_calibration':len(y_fit),'validation':len(y_val),'test':len(y_test)},
      'calibration':'3-fold cross-validation within fit pool (no validation or test data)',
      'selection':'lowest validation cost; tie-break Brier, then ROC-AUC; no test-driven selection',
      'cost_assumptions_brl':{'false_negative':default_cost,'false_positive':rejection_cost},
      'validation_candidates':validation,'selected_model':winner,'selected_threshold':threshold,
      'test':{'selected_threshold':result,'threshold_0_5':baseline,'bootstrap_95_ci':ci,
              'bootstrap_iterations':iterations},
      'limitations':['Small educational historical dataset; not production-ready',
                     'Bootstrap conditional on fixed trained model and threshold; does not capture training uncertainty',
                     'Cost figures hypothetical; no actual financial outcomes or fairness certification',
                     'Threshold optimized on validation; test evaluated once']}
    (output/'metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (output/'validation_candidates.csv').write_text(pd.DataFrame.from_dict(validation,orient='index').rename_axis('candidate').to_csv(),encoding='utf-8')
    fraction,mean=calibration_curve(y_test,p_test,n_bins=10,strategy='uniform')
    plt.figure(figsize=(6,5));plt.plot([0,1],[0,1],'--',label='Ideal');plt.plot(mean,fraction,'o-',label=winner)
    plt.xlabel('Mean predicted default probability');plt.ylabel('Observed default frequency')
    plt.title('Reliability diagram — held-out test');plt.legend();plt.tight_layout()
    plt.savefig(output/'calibration_curve.png',dpi=160);plt.close()
    plt.figure(figsize=(6,4));pd.DataFrame.from_dict(validation,orient='index')['cost_per_applicant'].sort_values().plot.bar()
    plt.ylabel('Validation hypothetical cost per applicant (BRL)');plt.title('Candidate selection — validation only')
    plt.tight_layout();plt.savefig(output/'validation_comparison.png',dpi=160);plt.close()
    print('V3 selected:',winner,'threshold:',threshold)
    print('Test ROC-AUC:',round(result['roc_auc'],4),'AP:',round(result['average_precision'],4),
          'Brier:',round(result['brier'],4),'Cost (BRL):',round(result['cost'],2))
    print('95% bootstrap intervals:',ci)
    print('Saved to',output.resolve())
    return payload

if __name__=='__main__': main()
