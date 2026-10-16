"""Leakage-aware model selection, cost-sensitive decisions and explainability."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_credit_data
from src.train import make_preprocessor

SEED = 42
DEFAULT_COST = 10000.0  # hypothetical loss for bad customer approved
REJECTION_COST = 1500.0  # hypothetical opportunity cost for good customer rejected


def business_cost(y_true, probabilities, threshold, default_cost=DEFAULT_COST, rejection_cost=REJECTION_COST):
    """Positive class 1=bad; approve when predicted probability is below threshold."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {'threshold': float(threshold), 'cost': float(fn * default_cost + fp * rejection_cost),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def find_best_threshold(y_true, probabilities, default_cost=DEFAULT_COST, rejection_cost=REJECTION_COST):
    """Optimize on validation data ONLY; ties favor lower threshold."""
    candidates = np.linspace(0.01, 0.99, 99)
    rows = [business_cost(y_true, probabilities, t, default_cost, rejection_cost) for t in candidates]
    return min(rows, key=lambda row: (row['cost'], row['threshold'])), pd.DataFrame(rows)


def build_candidates():
    models = {
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                                class_weight='balanced', random_state=SEED, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models['xgboost'] = XGBClassifier(n_estimators=180, max_depth=3, learning_rate=0.05,
                                          subsample=0.85, colsample_bytree=0.85,
                                          eval_metric='logloss', random_state=SEED, n_jobs=2)
    except ImportError:
        print('XGBoost unavailable; comparing logistic regression and random forest.')
    return models


def main(output_dir='outputs_v2', default_cost=DEFAULT_COST, rejection_cost=REJECTION_COST):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = load_credit_data()
    # 60% train / 20% validation / 20% untouched test
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25,
                                                      stratify=y_dev, random_state=SEED)
    candidates = build_candidates()
    validation_results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([('preprocessor', make_preprocessor(X_train)), ('model', estimator)])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        best, _ = find_best_threshold(y_val, probabilities, default_cost, rejection_cost)
        validation_results[name] = {'roc_auc': float(roc_auc_score(y_val, probabilities)),
                                    'average_precision': float(average_precision_score(y_val, probabilities)),
                                    'optimal_threshold': best['threshold'], 'validation_cost': best['cost']}
        fitted[name] = pipeline
        print(f'{name}: validation ROC-AUC={validation_results[name]["roc_auc"]:.3f}, '
              f'AP={validation_results[name]["average_precision"]:.3f}, '
              f'cost={best["cost"]:.0f}, threshold={best["threshold"]:.2f}')

    # Select using validation cost; break ties by validation ROC-AUC.
    winner = min(validation_results, key=lambda n: (validation_results[n]['validation_cost'],
                                                     -validation_results[n]['roc_auc']))
    threshold = validation_results[winner]['optimal_threshold']
    # Refit on development data, never touching test data during selection.
    final_model = Pipeline([('preprocessor', make_preprocessor(X_dev)),
                            ('model', build_candidates()[winner])])
    final_model.fit(X_dev, y_dev)
    p_test = final_model.predict_proba(X_test)[:, 1]
    selected = business_cost(y_test, p_test, threshold, default_cost, rejection_cost)
    baseline = business_cost(y_test, p_test, 0.5, default_cost, rejection_cost)
    preds = (p_test >= threshold).astype(int)
    results = {'dataset': 'UCI Statlog German Credit (id=144)', 'seed': SEED,
               'target': {'0': 'good', '1': 'bad'},
               'sample_counts': {'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)},
               'cost_assumptions_brl': {'false_negative': default_cost, 'false_positive': rejection_cost},
               'selection_rule': 'minimum validation cost, tie-break highest validation ROC-AUC',
               'validation_models': validation_results, 'selected_model': winner,
               'selected_threshold_from_validation': threshold,
               'test': {'roc_auc': float(roc_auc_score(y_test, p_test)),
                        'average_precision': float(average_precision_score(y_test, p_test)),
                        'threshold_0_5': baseline, 'selected_threshold': selected,
                        'classification_report': classification_report(y_test, preds, output_dict=True, zero_division=0)}}
    (out / 'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    joblib.dump(final_model, out / 'model.joblib')
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds, labels=[0, 1]), display_labels=['good', 'bad']).plot(cmap='Blues')
    plt.title(f'Test confusion matrix: {winner} (threshold={threshold:.2f})')
    plt.tight_layout(); plt.savefig(out / 'confusion_matrix.png', dpi=160); plt.close()
    fpr, tpr, _ = roc_curve(y_test, p_test)
    plt.figure(); plt.plot(fpr, tpr, label=f'{winner} (AUC={results["test"]["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], '--', label='Random'); plt.xlabel('False positive rate')
    plt.ylabel('True positive rate'); plt.title('ROC curve — held-out test'); plt.legend()
    plt.tight_layout(); plt.savefig(out / 'roc_curve.png', dpi=160); plt.close()
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    plt.figure(); plt.plot(recall, precision); plt.xlabel('Recall (bad)'); plt.ylabel('Precision (bad)')
    plt.title('Precision–Recall — held-out test'); plt.tight_layout()
    plt.savefig(out / 'precision_recall_curve.png', dpi=160); plt.close()
    _, curve = find_best_threshold(y_val, fitted[winner].predict_proba(X_val)[:, 1], default_cost, rejection_cost)
    plt.figure(); plt.plot(curve['threshold'], curve['cost']); plt.axvline(threshold, linestyle='--', label='Chosen on validation')
    plt.xlabel('Probability threshold for bad risk'); plt.ylabel('Hypothetical validation cost (BRL)')
    plt.title('Cost-sensitive threshold selection — validation only'); plt.legend(); plt.tight_layout()
    plt.savefig(out / 'validation_cost_curve.png', dpi=160); plt.close()
    curve.to_csv(out / 'validation_costs.csv', index=False)
    pd.DataFrame({'actual_bad': y_test.to_numpy(), 'predicted_probability_bad': p_test,
                  'predicted_bad': preds}).to_csv(out / 'test_predictions.csv', index=False)
    # SHAP on a small sample: tree or linear explainer via transformed feature matrix.
    try:
        import shap
        transformer = final_model.named_steps['preprocessor']
        estimator = final_model.named_steps['model']
        transformed = transformer.transform(X_test.iloc[:60])
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        feature_names = transformer.get_feature_names_out()
        if winner == 'logistic_regression':
            explainer = shap.LinearExplainer(estimator, transformed)
        else:
            explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(out / 'shap_summary.png', dpi=160, bbox_inches='tight'); plt.close()
        results['shap_status'] = 'generated'
    except Exception as exc:
        results['shap_status'] = f'unavailable: {type(exc).__name__}: {exc}'
        print('SHAP plot unavailable:', exc)
    (out / 'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSelected: {winner}; threshold={threshold:.2f}; test AUC={results["test"]["roc_auc"]:.3f}')
    print(f'Test cost @0.50: BRL {baseline["cost"]:,.0f}; @selected: BRL {selected["cost"]:,.0f}')
    print(f'Artifacts: {out.resolve()}')
    return results


if __name__ == '__main__':
    main()
