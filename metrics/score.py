import pandas as pd
import sklearn.metrics as ms

def model_metrics(y_true, y_pred, model, type='classification'):
    # type : 'classification', 'regression'
    y_true = y_true.values if isinstance(y_true, pd.DataFrame | pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.DataFrame | pd.Series) else y_pred
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if type == 'classification':
        n_class = len(set(y_true))
        return {
            'accuracy': ms.accuracy_score(y_true, y_pred),
            'balanced_accuracy': ms.balanced_accuracy_score(y_true, y_pred),
            'precision': ms.precision_score(y_true, y_pred, average='macro' if n_class > 2 else 'binary'),
            'recall': ms.recall_score(y_true, y_pred, average='macro' if n_class > 2 else 'binary'),
            'f1': ms.f1_score(y_true, y_pred, average='macro' if n_class > 2 else 'binary'),
            'roc_auc': ms.roc_auc_score(y_true, y_pred, average='macro' if n_class > 2 else 'binary'),
            'confusion_matrix': pd.DataFrame(ms.confusion_matrix(y_true, y_pred,labels=list(range(n_class)))),
            'classification_report': ms.classification_report(y_true, y_pred)
        }
    elif type == 'regression':
        return {
            'explained_variance_score': ms.explained_variance_score(y_true, y_pred),
            'max_error': ms.max_error(y_true, y_pred),
            'mean_absolute_error': ms.mean_absolute_error(y_true, y_pred),
            'mean_squared_error': ms.mean_squared_error(y_true, y_pred),
            # 'mean_squared_log_error': ms.mean_squared_log_error(y_true, y_pred),
            'median_absolute_error': ms.median_absolute_error(y_true, y_pred),
            'r2_score': ms.r2_score(y_true, y_pred),
        }