import gc
import numpy as np

import sklearn.feature_selection as fs
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import random
import copy
from model.unsupervised import HDBSCAN

class CVSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.__fetures_names = None
        self.__fetures_names_out = None

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        cols = X.columns.tolist()
        if len(cols) == 0:
            return self
        self.cv = (X.std() / X.mean()).abs()
        self.__fetures_names = set(X.columns.tolist())
        self.__fetures_names_out = set(self.cv.loc[self.cv > self.threshold].index.tolist())
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        cols = X.columns.tolist()
        if len(cols) == 0:
            return X if y is None else (X, y)
        assert set(X.columns.tolist()) == self.__fetures_names, 'columns must be same as fit'
        res = X.loc[:, list(self.__fetures_names_out)]
        return (res, y) if y is not None else res

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.cv = None
        self.__fetures_names = None
        self.__fetures_names_out = None
        gc.collect()


class NASelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = X.copy()
        data = data.to_frame() if isinstance(data, pd.Series) else data
        cols = data.columns.tolist()
        if len(cols) == 0:
            return data if y is None else (data.iloc[:, :-1], data.iloc[:, -1])
        self.threshold_feature = 1 - self.threshold if self.threshold else 1
        data = data.dropna(axis=1, thresh=self.threshold_feature * data.shape[0])
        if y is not None:
            return data, y
        else:
            return data

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
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
        if len(cols) == 0:
            return pd.DataFrame(columns=cols, index=['feature_importances_'])
        self.discrete_features_index = self.discrete_features
        if self.discrete_features not in ('auto', True, False):
            if isinstance(self.discrete_features, list):
                self.discrete_features_index = [cols.index(i) for i in self.discrete_features]
            else:
                raise ValueError('discrete_features must be auto, True, False or list')
        __score = self.__estimator(X, y, discrete_features=self.discrete_features_index,
                                        n_neighbors=self.n_nerighbors, random_state=1, *self.args, **self.kwargs)
        score = pd.DataFrame(__score.reshape(1, -1), columns=cols, index=['feature_importances_'])
        return score

    def fit(self, X, y):
        random.seed(self.random_state)
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        if len(X.columns.tolist()) == 0:
            return self
        res = [self.__fit(X, y) for _ in range(self.num_epochs)]
        self.score = pd.concat(res, axis=0).mean(axis=0)
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        cols = X.columns.tolist()
        if len(cols) == 0:
            return X if y is None else (X, y)
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
        self.info = None
        self.category_cols = None

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return self
        X_nunique = X.nunique()
        self.info = pd.concat([X_nunique, X.dtypes], axis=1)
        self.info.columns = ['n_unique', 'dtypes']
        self.info['dtypes'] = self.info['dtypes'].apply(lambda x: x.name)
        if 0 <= self.threshold < 1:
            self.category_cols = X_nunique.loc[X_nunique <= self.threshold * X.shape[0]].index.tolist()
        elif self.threshold >= 1 and int(self.threshold) == self.threshold:
            self.category_cols = X_nunique.loc[X_nunique <= self.threshold].index.tolist()
        else:
            raise ValueError
        res_cat_cols = self.info.loc[self.info['dtypes'].isin(['category', 'object'])].index.tolist()
        self.category_cols = list(set(self.category_cols) | set(res_cat_cols))
        return self

    def transform(self, X, y=None):
        assert self.info is not None, 'please fit first'
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        assert set(X_cols) >= set(self.info.index.tolist()), 'X columns must contain all fitted category columns'
        if not self.drop:
            X_cate = X[self.category_cols].astype(str)
        else:
            X_cate = X.drop(self.category_cols, axis=1)
        X_cate = X_cate.to_frame() if isinstance(X_cate, pd.Series) else X_cate
        return (X_cate, y) if y is not None else X_cate

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__cache = None
        self.info = None
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
        self.score = None

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        self.score = pd.Series(index=X_cols, name='MIC score')
        if len(X_cols) == 0:
            return self
        try:
            for cols in X_cols:
                self.model.compute_score(X[cols].values, y.values.ravel())
                self.score[cols] = self.model.mic()
        except Exception as e:
            self.score = None
            raise e
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        if self.score is None:
            raise ValueError('score is None, please fit first')
        if 0 <= self.threshold <= 1:
            mask = self.score.loc[self.score > self.threshold].index.tolist()
            return (X.loc[:, mask], y) if y is not None else X.loc[:, mask]
        elif self.threshold in range(1, X.shape[1] + 1):
            mask = self.score.sort_values(ascending=False).index[:self.threshold].values
            return (X.loc[:, mask], y) if y is not None else X.loc[:, mask]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.score = None
        gc.collect()


class DcorSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.score = None

    def distcorr(self, X, Y):
        from scipy.spatial.distance import pdist, squareform
        import numpy as np
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        if np.prod(X.shape) == len(X):
            X = X[:, None]
        if np.prod(Y.shape) == len(Y):
            Y = Y[:, None]
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        if Y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples must match')
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor

    def distcorr_broad(self, x, y):
        import numpy as np
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        assert len(x.shape) == 2 and len(y.shape) == 2, 'x and y must be 2d array'
        assert x.shape[1] == y.shape[1], 'x and y must have same dim at 1'
        import numpy as np
        dcor_matrix = [[self.distcorr(x[i, :], y[j, :]) for j in range(y.shape[0])]
                       for i in range(x.shape[0])]
        return np.array(dcor_matrix)


    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return self
        self.score = pd.Series(index=X_cols, name='DCOR score')
        try:
            for cols in X_cols:
                self.score[cols] = self.distcorr(X[cols].values, y.values.ravel())
        except Exception as e:
            self.score = None
            raise e
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        if self.score is None:
            raise ValueError('score is None, please fit first')
        if 0 <= self.threshold <= 1:
            mask = self.score.loc[self.score > self.threshold].index.tolist()
            return (X.loc[:, mask], y) if y is not None else X.loc[:, mask]
        elif self.threshold in range(1, X.shape[1] + 1):
            mask = self.score.sort_values(ascending=False).index[:self.threshold].values
            return (X.loc[:, mask], y) if y is not None else X.loc[:, mask]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.score = None
        gc.collect()


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, remain_y=True):
        self.cols = [cols] if (not isinstance(cols, list)) and (cols is not None) else cols
        self.remain_y = remain_y
        self.__is_fitted = False

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return self
        self.cols = X.columns.tolist() if self.cols is None else self.cols
        if y is not None:
            y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
            all_cols = X.columns.tolist() + y.columns.tolist()
            if set(self.cols) <= set(all_cols):
                self.__is_fitted = True
                return self
            else:
                raise ValueError('cols must be in X and y')
        else:
            all_cols = X.columns.tolist()
            if set(self.cols) <= set(all_cols):
                self.__is_fitted = True
                return self
            else:
                raise ValueError('cols must be in X')

    def transform(self, X, y=None):
        if not self.__is_fitted:
            raise ValueError('please fit first')
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        if y is not None:
            y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
            all_cols = X.columns.tolist() + y.columns.tolist()
            assert set(self.cols) <= set(all_cols), 'cols must be in X and y'
            y_cols = list(set(y.columns.tolist()) & set(self.cols))
            if len(y_cols) != 0:
                cols = list(set(self.cols) - set(y.columns.tolist()))
                return X.loc[:, cols], y[y_cols]
            else:
                return X.loc[:, self.cols] if not self.remain_y else (X.loc[:, self.cols], y)
        else:
            all_cols = X.columns.tolist()
            assert set(self.cols) <= set(all_cols), 'cols must be in X'
            return X.loc[:, self.cols]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__is_fitted = False
        gc.collect()


class IVSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.02):
        self.threshold = threshold

    def calculate_iv(self, x, y, type='category'):
        assert type in ['category', 'continuous'], 'type must be category or continuous'
        if type == 'continuous':
            cluster = HDBSCAN(min_cluster_size=5,
                              min_samples=None,
                              metric='euclidean',
                              cluster_selection_method='leaf')
            cluster.fit(x.reshape(-1, 1))
            x = cluster.labels_
        x, y = x.ravel(), y.ravel()

        # 计算每个分箱或分类中的正例和负例数量
        df = pd.DataFrame({'x': x, 'y': y})
        grouped = df.groupby('x')['y'].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']

        # 计算正例和负例的总数
        total_bad = grouped['bad'].sum()
        total_good = grouped['good'].sum()

        # 计算正例和负例的比率
        grouped['bad_rate'] = grouped['bad'] / total_bad
        grouped['good_rate'] = grouped['good'] / total_good

        # 避免除以0和对0取对数
        grouped['bad_rate'] = grouped['bad_rate'].replace(0, 0.00001)
        grouped['good_rate'] = grouped['good_rate'].replace(0, 0.00001)

        # 计算 IV
        grouped['iv'] = (grouped['bad_rate'] - grouped['good_rate']) * np.log(
            grouped['bad_rate'] / grouped['good_rate'])
        iv = grouped['iv'].sum()

        return iv

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        y_unique = y.nunique().iloc[0]
        if y_unique != 2:
            raise ValueError('y must be binary')
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return self
        self.score = pd.Series(index=X_cols, name='IV score')
        try:
            for cols in X_cols:
                self.score[cols] = self.calculate_iv(X[cols].values, y.values.ravel())
        except Exception as e:
            self.score = None
            raise e
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        if self.score is None:
            raise ValueError('score is None, please fit first')
        if isinstance(self.threshold, float):
            mask = self.score.loc[self.score > self.threshold].index.tolist()
            return (X.loc[:, mask], y) if y is not None else X.loc[:, mask]
        elif self.threshold in range(1, X.shape[1] + 1):
            mask = self.score.sort_values(ascending=False).index[:self.threshold].values
            return (X.loc[:, mask], y) if y is not None else X.loc[:, mask]
        else:
            raise ValueError('threshold must be integer in range(1, X.shape[1] + 1) or float between 0-1')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.score = None
        gc.collect()


class Chi2Selector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else y.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return self
        self.score = pd.DataFrame(index=X_cols, columns=['chi2', 'pvalue'])
        try:
            for cols in X_cols:
                self.score.loc[cols] = fs.chi2(X[cols].values.reshape(-1, 1), y.values.reshape(-1, 1))
        except Exception as e:
            self.score = None
            raise e
        return self

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        X_cols = X.columns.tolist()
        if len(X_cols) == 0:
            return X if y is None else (X, y)
        if self.score is None:
            raise ValueError('score is None, please fit first')
        mask = self.score.loc[self.score.pvalue < self.threshold].index.tolist()
        return (X.loc[:, mask], y) if y is not None else X.loc[:, mask]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.score = None
        gc.collect()
