import pandas as pd

class FeatureImportances:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def feature_importances(self, sorted=True, n_repeats=1, random_state=1, return_mean=True):
        res = []
        import numpy as np
        np.random.seed(random_state)
        for i in range(n_repeats):
            feature_importances = self.model.feature_importances_ if hasattr(self.model,
                                                                             'feature_importances_') else None
            if feature_importances is not None:
                feature_importances = pd.Series(data=feature_importances, index=self.X.columns.tolist(),
                                                name='feature_importances')
            re = feature_importances / feature_importances.sum()
            res.append(re.to_frame().T)
        res = pd.concat(res, axis=0)
        if return_mean:
            res = res.mean(axis=0)
            if sorted:
                res = res.sort_values(ascending=False)
        return res / sum(res)

    def permutation_importances(self, n_repeats=100, sorted=False, random_state=1, n_jobs=None):
        from sklearn.inspection._permutation_importance import permutation_importance
        import multiprocessing
        n_jobs = multiprocessing.cpu_count() if n_jobs is None else n_jobs
        result = permutation_importance(self.model, self.X, self.y, n_repeats=n_repeats, random_state=random_state,
                                        n_jobs=n_jobs)
        feature_importances = pd.Series(data=result.importances_mean, index=self.X.columns.tolist(),
                                        name='feature_importances')
        if sorted:
            feature_importances = feature_importances.sort_values(ascending=False)
        return feature_importances / feature_importances.sum()

    def shap_value(self, model_type='tree', sorted=False, return_mean=True):
        import shap
        explainer = shap.TreeExplainer(self.model) if model_type == 'tree' else shap.KernelExplainer(self.model.predict,
                                                                                                     self.X)
        shap_values = explainer.shap_values(self.X)[0]
        shap_values = pd.DataFrame(shap_values, columns=self.X.columns) if isinstance(self.X,
                                                                                      pd.DataFrame) else shap_values
        if return_mean:
            shap_values = shap_values.abs().mean(axis=0)
            if sorted:
                shap_values = shap_values.sort_values(ascending=False)
        return shap_values / shap_values.sum()

    def null_importances(self, n_repeats=100, random_state=1, quantile=0.75, sorted=False):
        null_importances = []
        real_importances = []
        import numpy as np
        np.random.seed(random_state)
        for i in range(n_repeats):
            y_permuted = self.y.copy().sample(frac=1.0)
            self.model.fit(self.X, y_permuted)
            result = self.feature_importances(sorted=False).to_frame().transpose()
            null_importances.append(result)
            self.model.fit(self.X, self.y)
            result = self.feature_importances(sorted=False).to_frame().transpose()
            real_importances.append(result)
        null_importances = pd.concat(null_importances, axis=0, ignore_index=True)
        real_importances = pd.concat(real_importances, axis=0, ignore_index=True)
        real_importances = real_importances.mean(axis=0)
        real_importances = real_importances / real_importances.sum()
        null_importances = null_importances.apply(lambda x: x.quantile(quantile), axis=0)
        null_importances = null_importances / null_importances.sum()
        res = (real_importances / null_importances)
        if sorted:
            res = res.sort_values(ascending=False)
        return res / res.sum()

    def report(self):
        default = self.feature_importances(sorted=False)
        permutation = self.permutation_importances(sorted=False)
        shap = self.shap_value(sorted=False)
        null = self.null_importances(sorted=False)
        report = pd.concat([default, permutation, shap, null], axis=1)
        report.columns = ['default', 'permutation', 'shap', 'null']
        return report
