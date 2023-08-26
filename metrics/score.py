import numpy as np
import pandas as pd
import sklearn.metrics as ms

def get_score(y_true, y_pred, y_proba=None, type='classification'):
    # type : 'classification', 'regression'
    y_true = y_true.values if isinstance(y_true, pd.DataFrame | pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.DataFrame | pd.Series) else y_pred
    y_true = y_true.ravel() if isinstance(y_true, np.ndarray) else y_true
    y_pred = y_pred.ravel() if isinstance(y_pred, np.ndarray) else y_pred
    if type == 'classification':
        n_class = len(set(y_true))
        results = {
            'accuracy': ms.accuracy_score(y_true, y_pred),
            'balanced_accuracy': ms.balanced_accuracy_score(y_true, y_pred),
            'precision': ms.precision_score(y_true, y_pred, average='macro' if n_class > 2 else 'binary'),
            'recall': ms.recall_score(y_true, y_pred, average='macro' if n_class > 2 else 'binary'),
            'f1': ms.f1_score(y_true, y_pred, average='macro' if n_class > 2 else 'binary'),
            'confusion_matrix': pd.DataFrame(ms.confusion_matrix(y_true, y_pred,labels=list(range(n_class)))),
            'classification_report': ms.classification_report(y_true, y_pred)
        }
        if y_proba is not None:
            y_proba = y_proba.values if isinstance(y_proba, pd.DataFrame | pd.Series) else y_proba
            results['roc_auc'] = ms.roc_auc_score(y_true, y_proba, average='macro' if n_class > 2 else 'binary')
    elif type == 'regression':
        results = {
            'explained_variance_score': ms.explained_variance_score(y_true, y_pred),
            'max_error': ms.max_error(y_true, y_pred),
            'mean_absolute_error': ms.mean_absolute_error(y_true, y_pred),
            'mean_squared_error': ms.mean_squared_error(y_true, y_pred),
            # 'mean_squared_log_error': ms.mean_squared_log_error(y_true, y_pred),
            'median_absolute_error': ms.median_absolute_error(y_true, y_pred),
            'r2_score': ms.r2_score(y_true, y_pred),
        }
    return results
