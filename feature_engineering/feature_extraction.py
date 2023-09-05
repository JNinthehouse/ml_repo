import gc
import numpy as np
import openfe
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import copy
import sklearn.manifold as smd

class FeatureBoostingGenerator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_jobs=None,
                 task='classification',
                 numerical_features=None,
                 categorical_features=None,
                 ordinal_features=None,
                 verbose=False,
                 seed=1,
                 feature_boosting=True,
                 stage1_metric='predictive',
                 stage2_metric='permutation',
                 min_candidate_features=500
                 ):
        self._generator = openfe.OpenFE()
        assert task in ['classification', 'regression'], 'task must be classification or regression'
        self.task = task
        self.get_formula = openfe.tree_to_formula
        if n_jobs is not None:
            self.n_jobs = n_jobs
        else:
            import multiprocessing
            self.n_jobs = multiprocessing.cpu_count()
        self.new_features = None
        self.num_features = [numerical_features] if (not isinstance(numerical_features, list)) and (
                    numerical_features is not None) else numerical_features
        self.cat_features = [categorical_features] if (not isinstance(categorical_features, list)) and (
                    categorical_features is not None) else categorical_features
        self.ord_features = [ordinal_features] if (not isinstance(ordinal_features, list)) and (
                    ordinal_features is not None) else ordinal_features
        self.verbose = verbose
        self.feature_boosting = feature_boosting
        self.seed = seed
        self.stage1_metric = stage1_metric
        self.stage2_metric = stage2_metric
        self.min_candidate_features = min_candidate_features

    def fit(self, X, y):
        self.candidate_features = openfe.get_candidate_features(numerical_features=self.num_features,
                                                                categorical_features=self.cat_features,
                                                                ordinal_features=self.ord_features)
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return self
        target_cols = []
        target_cols = target_cols + self.num_features if self.num_features is not None else target_cols
        target_cols = target_cols + self.cat_features if self.cat_features is not None else target_cols
        target_cols = target_cols + self.ord_features if self.ord_features is not None else target_cols
        assert set(target_cols) <= set(X_cols), 'target_cols must be subset of X.columns'
        self.new_features = self._generator.fit(data=X,
                                                label=y,
                                                task=self.task,
                                                candidate_features_list=self.candidate_features,
                                                verbose=self.verbose,
                                                n_jobs=self.n_jobs,
                                                seed=self.seed,
                                                stage1_metric=self.stage1_metric,
                                                stage2_metric=self.stage2_metric,
                                                feature_boosting=self.feature_boosting,
                                                min_candidate_features=self.min_candidate_features
                                                )
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        data, _ = self._generator.transform(X, X, self.new_features, n_jobs=self.n_jobs)
        if len(self.new_features) != 0:
            X = data.iloc[:, :-len(self.new_features)]
            new_data = data.iloc[:, -len(self.new_features):]
            new_data.columns = ['NF_' + str(i) for i in range(len(self.new_features))]
            data = pd.concat([X, new_data], axis=1)
        return (data, y) if y is not None else data

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    @property
    def formula(self):
        feature_formula = {'NF_' + str(i): openfe.tree_to_formula(self.new_features[i]) for i in
                           range(len(self.new_features))}
        return feature_formula

    def clear(self):
        self.new_features = None
        self.num_features = None
        self.cat_features = None
        self.ord_features = None
        self.candidate_features = None
        gc.collect()


class CatBoostEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cols=None,
                 drop_invariant=False,
                 return_df=True,
                 handle_missing='value',
                 handle_unknown='value',
                 random_state=None,
                 sigma=None,
                 a=1):
        # handel_missing:'error', np.nan, 'value'
        # handel_unknown:'error', np.nan, 'value'
        import category_encoders as ce
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_missing = 'return_nan' if handle_missing != handle_missing else handle_missing
        self.handle_unknown = 'return_nan' if handle_unknown != handle_unknown else handle_unknown
        self.random_state = random_state
        self.sigma = sigma
        self.a = a
        self._mapping = None
        self._ce = ce.CatBoostEncoder(cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                                      handle_missing=handle_missing, handle_unknown=handle_unknown,
                                      random_state=random_state, sigma=sigma, a=a)

    def fit(self, X, y):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return self
        if self.cols is None or len(self.cols) == 0:
            self.cols = X.columns.tolist()
        self._ce.fit(X[self.cols], y)
        return self

    def clear(self):
        self.cols = None
        self._mapping = None
        gc.collect()

    @property
    def cols(self):
        return self._ce.cols

    @cols.setter
    def cols(self, cols):
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self._ce.cols = cols

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        X[self.cols] = X[self.cols].astype(float)
        X[self.cols] = self._ce.transform(X[self.cols])
        return (X, y) if y is not None else X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class OrdinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None,
                 handle_unknown='value',
                 unknown_value=np.nan,
                 handle_missing=np.nan,
                 min_frequency=1,
                 ):
        # handle_unknown: 'error', 'value'
        # handle_missing: np.nan, int
        from sklearn.preprocessing import OrdinalEncoder as oe
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.cols = cols
        self.handle_unknown = 'use_encoded_value' if handle_unknown == 'value' else 'error'
        self.unknown_value = unknown_value
        self.missing_value = handle_missing
        self.min_frequency = min_frequency
        self._oe = oe(handle_unknown=self.handle_unknown,
                      unknown_value=self.unknown_value,
                      encoded_missing_value=self.missing_value,
                      min_frequency=self.min_frequency)

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        if self.cols is None:
            self.cols = X.columns.tolist()
        if len(X.columns.tolist()) == 0:
            return self
        self._oe.fit(X[self.cols].values.reshape(-1, len(self.cols)))
        return self

    def clear(self):
        self.cols = None
        gc.collect()

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        if X.shape[1] == 0:
            return X if y is None else (X, y)
        X[self.cols] = self._oe.transform(X[self.cols].values.reshape(-1, len(self.cols)))
        X[self.cols] = X[self.cols].astype(float)
        return (X, y) if y is not None else X

    def inverse_transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        if X.shape[1] == 0:
            return X if y is None else (X, y)
        X[self.cols] = self._oe.inverse_transform(X[self.cols].values.reshape(-1, len(self.cols)))
        return (X, y) if y is not None else X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None,
                 handle_unknown=np.nan,
                 handle_missing=np.nan,
                 min_group_size=1,
                 normalize=True,
                 ):
        # handle_unknown: 'error', 'value', np.nan
        # handle_missing: 'error', 'value', np.nan
        import category_encoders as ce
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.handle_unknown = 'return_nan' if handle_unknown == np.nan else handle_unknown
        self.handle_missing = 'return_nan' if handle_missing == np.nan else handle_missing
        self.min_group_size = min_group_size
        self.normalize = normalize
        self._mapping = None
        self._ce = ce.CountEncoder(cols=cols,
                                   handle_unknown=self.handle_unknown,
                                   handle_missing=self.handle_missing,
                                   min_group_size=self.min_group_size,
                                   normalize=self.normalize)

    def clear(self):
        self.cols = None
        self._mapping = None
        gc.collect()

    @property
    def cols(self):
        return self._ce.cols

    @cols.setter
    def cols(self, cols):
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self._ce.cols = cols

    def fit(self, X, y=None):
        data = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        if X.shape[1] == 0:
            return self
        if self.cols is None:
            self.cols = X.columns.tolist()
        self._ce.fit(X[self.cols].astype(str))
        self._mapping = self._ce.mapping
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        if X.shape[1] == 0:
            return X if y is None else (X, y)
        X[self.cols] = self._ce.transform(X[self.cols])
        X[self.cols] = X[self.cols].astype(float)
        return (X, y) if y is not None else X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        if X.shape[1] == 0:
            return X if y is None else (X, y)
        for key, value in self._mapping.items():
            value = value.to_dict()
            value = {v: k for k, v in value.items()}
            X[key] = X[key].map(value)
        return (X, y) if y is not None else X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None,
                 use_cat_names=True,
                 handle_unknown='value',
                 handle_missing='value',
                 ):
        # handle_unknown: 'error', 'value', np.nan
        # handle_missing: 'error', 'value', np.nan, 'indicator'
        from category_encoders import OneHotEncoder as ohe
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.handle_unknown = 'return_nan' if handle_unknown == np.nan else handle_unknown
        self.handle_missing = 'return_nan' if handle_missing == np.nan else handle_missing
        self.use_cat_names = use_cat_names
        self._mapping = None
        self._ohe = ohe(cols=cols,
                        handle_unknown=self.handle_unknown,
                        handle_missing=self.handle_missing,
                        use_cat_names=self.use_cat_names)

    @property
    def cols(self):
        return self._ohe.cols

    @cols.setter
    def cols(self, cols):
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self._ohe.cols = cols

    def fit(self, X, y=None):
        data = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        self.cols = data.columns.tolist() if self.cols is None else self.cols
        if data.shape[1] == 0:
            return self
        self._ohe.fit(data[self.cols].astype('category'))
        self._mapping = self._ohe.mapping
        return self

    def transform(self, X, y=None):
        data = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        cur_cols = data.columns.tolist()
        if len(cur_cols) == 0:
            return data if y is None else (data, y)
        res_cols = list(set(cur_cols) - set(self.cols))
        data_encoded = self._ohe.transform(data[self.cols])
        self.transformed_cols = data_encoded.columns.tolist()
        if len(res_cols) > 0:
            data_encoded = pd.concat([data[res_cols], data_encoded], axis=1)
        return (data_encoded, y) if y is not None else data_encoded

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        data = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        cur_cols = data.columns.tolist()
        if len(cur_cols) == 0:
            return data if y is None else (data, y)
        res_cols = list(set(cur_cols) - set(self.transformed_cols))
        data_encoded = self._ohe.inverse_transform(data[self.transformed_cols])
        if len(res_cols) > 0:
            data_encoded = pd.concat([data[res_cols], data_encoded], axis=1)
        return (data_encoded, y) if y is not None else data_encoded

    def clear(self):
        self.__cache = None
        self._mapping = None
        self.cols = None
        self.transformed_cols = None
        gc.collect()


class WOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cols=None,
                 handle_unknown='value',
                 handle_missing=np.nan,
                 randomized=False,
                 sigma=0.05,
                 ):
        # handle_unknown: 'error', 'value', np.nan
        # handle_missing: 'error', 'value', np.nan
        from category_encoders import WOEEncoder as we
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.handle_unknown = 'return_nan' if handle_unknown == np.nan else handle_unknown
        self.handle_missing = 'return_nan' if handle_missing == np.nan else handle_missing
        self.randomized = randomized
        self.sigma = sigma
        self.__we = we(cols=cols,
                       handle_unknown=self.handle_unknown,
                       handle_missing=self.handle_missing,
                       randomized=self.randomized,
                       sigma=self.sigma)

    @property
    def cols(self):
        return self.__we.cols

    @cols.setter
    def cols(self, cols):
        cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.__we.cols = cols

    def fit(self, X, y):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        if len(X.columns.tolist()) == 0:
            return self
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        y_value_counts = y.iloc[:, 0].value_counts()
        if len(y_value_counts) != 2:
            raise ValueError('WOEEncoder only support binary classification')
        else:
            y_values_map = {y_value_counts.index[0]: 0, y_value_counts.index[1]: 1}
            y.iloc[:, 0] = y.iloc[:, 0].map(y_values_map)
            X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
            if self.cols is None:
                self.cols = X.columns.tolist()
            else:
                if set(self.cols) <= set(X.columns.tolist()):
                    pass
                else:
                    raise ValueError('cols must be subset of X.columns')
            self.__we.fit(X[self.cols], y.iloc[:, 0])
            return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        if len(X.columns.tolist()) == 0:
            return X if y is None else (X, y)
        X[self.cols] = self.__we.transform(X[self.cols])
        return (X, y) if y is not None else X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        gc.collect()


class PCAGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_components=10, random_state=1, kernel='linear', gamma=None, degree=3, coef0=1,
                 n_jobs=-1):
        assert kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine',
                          'precomputed'], 'kernel must be one of linear, poly, rbf, sigmoid, cosine, precomputed'
        from sklearn.decomposition import KernelPCA
        self.__estimator = KernelPCA(n_components=None,
                                     random_state=random_state,
                                     kernel=kernel,
                                     gamma=gamma,
                                     degree=degree,
                                     coef0=coef0,
                                     n_jobs=n_jobs)
        self.cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.n_components = n_components
        self.var_ratio = None

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        if self.cols is None:
            self.cols = X.columns.tolist()
        if len(X.columns.tolist()) == 0:
            return self
        self.__estimator.fit(X[self.cols])
        self.var_ratio = self.__estimator.eigenvalues_ / self.__estimator.eigenvalues_.sum()
        self.var_ratio = pd.Series(data=self.var_ratio, index=self.__estimator.get_feature_names_out().tolist(),
                                   name='exp_var_ratio')
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        if len(X.columns.tolist()) == 0:
            return X if y is None else (X, y)
        y = y.to_frame() if isinstance(y, pd.Series) else copy.copy(y)
        assert set(self.cols) <= set(X.columns.tolist()), 'cols must be subset of X.columns'
        not_fitted_cols = list(set(X.columns.tolist()) - set(self.cols))
        res = self.__estimator.transform(X[self.cols])
        res = pd.DataFrame(res)
        res = res.iloc[:, :self.n_components]
        res.columns = self.var_ratio.index.tolist()[:self.n_components]
        X = pd.concat([X[not_fitted_cols], res], axis=1)
        return (X, y) if y is not None else X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.var_ratio = None
        gc.collect()

    @property
    def random_state(self):
        return self.__estimator.random_state

    @random_state.setter
    def random_state(self, random_state):
        self.__estimator.random_state = random_state

    @property
    def kernel(self):
        return self.__estimator.kernel

    @kernel.setter
    def kernel(self, kernel):
        self.__estimator.kernel = kernel

    @property
    def gamma(self):
        return self.__estimator.gamma

    @gamma.setter
    def gamma(self, gamma):
        self.__estimator.gamma = gamma

    @property
    def degree(self):
        return self.__estimator.degree

    @degree.setter
    def degree(self, degree):
        self.__estimator.degree = degree

    @property
    def coef0(self):
        return self.__estimator.coef0

    @coef0.setter
    def coef0(self, coef0):
        self.__estimator.coef0 = coef0

    @property
    def n_jobs(self):
        return self.__estimator.n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self.__estimator.n_jobs = n_jobs
