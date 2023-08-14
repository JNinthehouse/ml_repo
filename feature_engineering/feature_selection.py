import gc
import sklearn.feature_selection as fs
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import random


class CVSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.__selector = fs.VarianceThreshold(threshold=self.threshold)

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold
        self.__selector = fs.VarianceThreshold(threshold=self.threshold)

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cv = (X.std() / X.mean()).values
        self.__selector = self.__selector.fit(X)
        self.__selector.variances_ = X_cv
        self.cv = pd.Series(X_cv, index=X.columns, name='Coefficient of Variation')
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        res = self.__selector.transform(X)
        res = pd.DataFrame(res, columns=self.__selector.get_feature_names_out())
        return (res, y) if y is not None else res

    def fit_transform(self, X, y=None, **fit_params):
        model = self.fit(X, y)
        return model.transform(X, y)

    def clear(self):
        self.__cache = None
        self.cv = None
        gc.collect()


class NASelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = pd.concat([X.copy(), y.copy()], axis=1) if y is not None else X.copy()
        data = data.to_frame() if isinstance(data, pd.Series) else data
        self.threshold_feature = 1 - self.threshold if self.threshold else 1
        data = data.dropna(axis=1, thresh=self.threshold_feature * data.shape[0])
        if y is not None:
            return data.iloc[:, :-1], data.iloc[:, -1]
        else:
            return data

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__cache = None
        self.threshold_feature = None
        gc.collect()


class MatualInfoSelector(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_nerighbors=5,
                 task='classification',
                 discrete_features='auto',
                 num_epochs=100,
                 random_state=1,
                 threshold=0.0,
                 *args, **kwargs):
        # discrete_features: {‘auto’, True, False} or list of feature names, default=’auto’
        self.n_nerighbors = n_nerighbors
        self.args = args
        self.kwargs = kwargs
        self.discrete_features = discrete_features
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.threshold = threshold
        if task == 'classification':
            self.__estimator = fs.mutual_info_classif
        elif task == 'regression':
            self.__estimator = fs.mutual_info_regression
        else:
            raise ValueError('task must be classification or regression')

    def __fit(self, X, y):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        cols = list(X.columns)
        self.discrete_features_index = self.discrete_features
        if self.discrete_features not in ('auto', True, False):
            if isinstance(self.discrete_features, list):
                self.discrete_features_index = [cols.index(i) for i in self.discrete_features]
            else:
                raise ValueError('discrete_features must be auto, True, False or list')
        self.__score = self.__estimator(X, y, discrete_features=self.discrete_features_index,
                                        n_neighbors=self.n_nerighbors, random_state=1, *self.args, **self.kwargs)
        self.score = pd.DataFrame(self.__score.reshape(1, -1), columns=cols, index=['feature_importances_'])
        return self.score

    def fit(self, X, y):
        random.seed(self.random_state)
        res = [self.__fit(X, y) for _ in range(self.num_epochs)]
        self.score = pd.concat(res, axis=0).mean(axis=0)
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        if 0 <= self.threshold <= 1:
            mask = (self.score > self.threshold).values.ravel()
            return (X.iloc[:, mask], y) if y is None else X.iloc[:, mask]
        elif self.threshold in range(1, X.shape[1] + 1):
            mask = self.score.sort_values(ascending=False).index[:self.threshold].values
            return (X.loc[:, mask], y) if y is None else X.loc[:, mask]
        else:
            raise ValueError(f'threshold must be integer in range(1, {X.shape[1] + 1}) or float between 0-1')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__score = None
        self.score = None
        self.discrete_features_index = None
        self.__cache = None
        gc.collect()


class CategorySelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10, drop=False):
        self.threshold = threshold
        self.drop = drop
        self.unique_values = None

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_nunique = X.nunique()
        self.unique_values = X_nunique
        if 0 <= self.threshold < 1:
            self.category_cols = X_nunique.loc[X_nunique <= self.threshold * X.shape[0]].index.tolist()
        elif self.threshold >= 1 and int(self.threshold) == self.threshold:
            self.category_cols = X_nunique.loc[X_nunique <= self.threshold].index.tolist()
        else:
            raise ValueError
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        cate_cols = list(set(X_cols) & set(self.category_cols))
        if not self.drop:
            X_cate = X[cate_cols].astype('category')
        else:
            X_cate = X.drop(cate_cols, axis=1)
        X_cate = X_cate.to_frame() if isinstance(X_cate, pd.Series) else X_cate
        return (X_cate, y) if y is not None else X_cate

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__cache = None
        self.unique_values = None
        self.category_cols = None
        gc.collect()


class MICSelector(BaseEstimator, TransformerMixin):
    # env:train_dev python==3.10
    def __init__(self, alpha=4, c=15, est='mic_approx', threshold=0.1):
        from minepy import MINE
        self.alpha = alpha
        self.c = c
        self.est = est
        self.model = MINE(alpha=alpha, c=c, est=est)
        self.threshold = threshold

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        self.model.compute_score(X.values, y.values)
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        self.score = pd.DataFrame(self.model.mic(), index=X_cols, columns=['mic'])
        if 0 <= self.threshold <= 1:
            mask = (self.score > self.threshold).values.ravel()
            return (X.iloc[:, mask], y) if y is None else X.iloc[:, mask]
        elif self.threshold in range(1, X.shape[1] + 1):
            mask = self.score.sort_values(ascending=False).index[:self.threshold].values
            return (X.loc[:, mask], y) if y is None else X.loc[:, mask]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)
