import pandas as pd
from sklearn.inspection._permutation_importance import permutation_importance
import shap
import sklearn.metrics as ms


def feature_importances(model, X):
    feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    if feature_importances:
        feature_importances = pd.DataFrame({'feature_names': X.columns, 'feature_importances': feature_importances})
        feature_importances = feature_importances.sort_values(by='feature_importances', ascending=False)
    return feature_importances


def permutation_importances(model, X, y, n_repeats=100, random_state=1, n_jobs=2):
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
    feature_importances = pd.DataFrame({'feature_names': X.columns, 'feature_importances': result.importances_mean})
    feature_importances = feature_importances.sort_values(by='feature_importances', ascending=False)
    return feature_importances


def shap_value(model, X, model_type='tree', return_mean=True):
    explainer = shap.TreeExplainer(model) if model_type == 'tree' else shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)[1]
    shap_values = pd.DataFrame(shap_values, columns=X.columns) if isinstance(X, pd.DataFrame) else shap_values
    if return_mean:
        shap_values = shap_values.abs().mean(axis=0)
    return shap_values


if __name__ == '__main__':
    import sklearn.datasets as ds

    data = ds.load_breast_cancer(as_frame=True)
    df = data['data']
    df['target'] = data['target']
    import lightgbm as lgb

    model = lgb.LGBMClassifier()
    model.fit(df.drop('target', axis=1), df['target'])
