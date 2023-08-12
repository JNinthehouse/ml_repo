import numpy as np
import sklearn.preprocessing as pre
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import gc

class DetectOutlier(BaseEstimator, TransformerMixin):
    def __init__(self, drop=True, cols=None, threshold=0.5, method='iqr', **kwargs):
        self.drop = drop
        self.cols = cols
        self.threshold = threshold
        self.method = method
        self.kwargs = kwargs

    def fit(self, X, y=None):
        data = X.copy()
        data = data.to_frame() if isinstance(data, pd.Series) else data
        cols = self.cols if self.cols is not None else data.columns
        if self.method == 'iqr':
            self.q1 = data[cols].quantile(0.25).to_frame().T
            self.q3 = data[cols].quantile(0.75).to_frame().T
            self.iqr = self.q3.reset_index(drop=True) - self.q1.reset_index(drop=True)
        elif self.method == 'std':
            self.mean = dict()
            self.std = dict()
            for col in cols:
                self.mean[col] = data[col].mean()
                self.std[col] = data[col].std()
        elif self.method == 'isoforest':
            from sklearn.ensemble import IsolationForest
            self.clf = IsolationForest(**self.kwargs)
            self.clf = self.clf.fit(data[cols])
        else:
            raise NotImplementedError
        self.__cache = data
        return self

    def transform(self, X, y=None):
        data = X.copy()
        data = data.to_frame() if isinstance(data, pd.Series) else data
        self.cols = self.cols if self.cols is not None else data.columns
        if set(self.cols) != set(self.__cache.columns):
            raise ValueError('The columns of X is not equal to the columns of fitted data')

        # 处理方法生成mask
        if self.method == 'iqr':
            self.mask = data[self.cols].values.reshape(-1, len(self.cols))
            self.mask = (self.mask < (self.q1.values - 1.5 * self.iqr.values)) | (
                    self.mask > (self.q3.values + 1.5 * self.iqr.values))
            self.mask = pd.DataFrame(self.mask, columns=self.cols, index=data.index)
            self.mask = self.mask.sum(axis=1) >= self.threshold * len(self.cols)
        if self.method == 'std':
            self.mask = pd.DataFrame(np.zeros_like(data), columns=self.cols, index=data.index)
            for col in self.cols:
                self.mask[col] = data[col].apply(
                    lambda x: True if abs(x - self.mean[col]) > 3 * self.std[col] else False)
            self.mask = self.mask.to_frame() if isinstance(self.mask, pd.Series) else self.mask
            self.mask = self.mask.sum(axis=1) >= self.threshold * len(self.cols)
        if self.method == 'isoforest':
            mask = (self.clf.predict(data[self.cols]) == -1)
            self.mask = pd.Series(mask, index=data.index)
        self.mask.name = 'is_outlier'

        # 处理数据
        if not self.drop:
            data = pd.concat([self.mask.map({True: 1, False: 0}), data], axis=1, ignore_index=True)
        else:
            data = data.loc[~self.mask]
        if y is not None:
            return data.iloc[:, :-1], data.iloc[:, -1]
        else:
            return data

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__cache = None
        self.mask = None
        gc.collect()


class FillNA(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, by=None, method='mean'):
        self.cols = cols
        self.by = by
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = self.cols if self.cols else list(X.columns)
        data = pd.concat([X.copy(), y.copy()], axis=1) if y is not None else X.copy()
        data = data.to_frame() if isinstance(data, pd.Series) else data

        if self.method == 'mean':
            data[cols] = data[cols].fillna(data[cols].mean()) if not self.by else data.groupby(self.by)[cols].transform(
                lambda x: x.fillna(x.mean()))
        elif self.method == 'median':
            data[cols] = data[cols].fillna(data[cols].median()) if not self.by else data.groupby(self.by)[
                cols].transform(lambda x: x.fillna(x.median()))
        elif self.method == 'mode':
            data[cols] = data[cols].fillna(data[cols].mode()[0]) if not self.by else data.groupby(self.by)[
                cols].transform(lambda x: x.fillna(x.mode()[0]))
        else:
            raise NotImplementedError

        if (y.isnull().any()) & (y is not None):
            print('Warning: target still has missing values')

        if y is not None:
            return data.iloc[:, :-1], data.iloc[:, -1]
        else:
            return data

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__cache = None
        gc.collect()


class DropNA(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, threshold=0.5):
        self.threshold_sample = threshold
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = pd.concat([X.copy(), y.copy()], axis=1) if y is not None else X.copy()
        data = data.to_frame() if isinstance(data, pd.Series) else data
        data = data.loc[data.iloc[:, -1].notnull()] if y is not None else data  # 去除y缺失的行
        self.threshold_sample = 1 - self.threshold_sample if self.threshold_sample else 1
        data = data.dropna(axis=0, thresh=self.threshold_sample * data.shape[1], subset=self.cols)
        if y is not None:
            return data.iloc[:, :-1], data.iloc[:, -1]
        else:
            return data

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__cache = None
        gc.collect()


class DropDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = pd.concat([X.copy(), y.copy()], axis=1) if y is not None else X.copy()
        data = data.to_frame() if isinstance(data, pd.Series) else data
        data = data.drop_duplicates() if self.cols is not None else data.drop_duplicates(subset=self.cols)
        if y is not None:
            return data.iloc[:, :-1], data.iloc[:, -1]
        else:
            return data

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def clear(self):
        self.__cache = None
        gc.collect()

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, method='robust'):
        self.cols = cols
        self.method = method

    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scaler = pre.StandardScaler()
        elif self.method == 'minmax':
            self.scaler = pre.MinMaxScaler()
        elif self.method == 'maxabs':
            self.scaler = pre.MaxAbsScaler()
        elif self.method == 'robust':
            self.scaler = pre.RobustScaler()
        else:
            raise NotImplementedError
        df_copy = X.copy()
        df_copy = df_copy.to_frame() if isinstance(df_copy, pd.Series) else df_copy
        self.cols = self.cols if self.cols is not None else list(df_copy.columns)
        self.scaler = self.scaler.fit(df_copy[self.cols])
        return self

    def transform(self, X, y=None):
        df_copy = X.copy()
        df_copy = df_copy.to_frame() if isinstance(df_copy, pd.Series) else df_copy
        df_copy[self.cols] = self.scaler.transform(df_copy[self.cols])
        if y is not None:
            return df_copy, y
        else:
            return df_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        df_copy = X.copy()
        df_copy = df_copy.to_frame() if isinstance(df_copy, pd.Series) else df_copy
        df_copy[self.cols] = self.scaler.inverse_transform(df_copy[self.cols])
        if y is not None:
            return df_copy, y
        else:
            return df_copy

    def clear(self):
        self.__cache = None
        gc.collect()


class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, method='yeo-johnson'):
        self.cols = cols
        self.method = method

    def fit(self, X, y=None):
        if self.method == 'yeo-johnson':
            self.normalizer = pre.PowerTransformer(method=self.method)
        elif self.method == 'box-cox':
            self.normalizer = pre.PowerTransformer(method=self.method)
        else:
            raise NotImplementedError
        df_copy = X.copy()
        df_copy = df_copy.to_frame() if isinstance(df_copy, pd.Series) else df_copy
        self.cols = self.cols if self.cols is not None else df_copy.columns
        self.normalizer = self.normalizer.fit(df_copy[self.cols])
        return self

    def transform(self, X, y=None):
        df_copy = X.copy()
        df_copy = df_copy.to_frame() if isinstance(df_copy, pd.Series) else df_copy
        df_copy[self.cols] = self.normalizer.transform(df_copy[self.cols])
        if y is not None:
            return df_copy, y
        else:
            return df_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        df_copy = X.copy()
        df_copy = df_copy.to_frame() if isinstance(df_copy, pd.Series) else df_copy
        df_copy[self.cols] = self.normalizer.inverse_transform(df_copy[self.cols])
        if y is not None:
            return df_copy, y
        else:
            return df_copy

    def clear(self):
        self.__cache = None
        gc.collect()


class Sampler(BaseEstimator, TransformerMixin):
    def __init__(self, method='borderlinesmote', **kwargs):
        # borderlinesmote 参数
        # sampling_strategy = 'auto', 自动均衡样本
        # random_state = 42, 随机种子
        # k_neighbors = 5, 近邻数
        # m_neighbors = 10, 近邻数，主要是为了计算密度区分是否为边界点
        # kind = 'borderline-1',
        # n_jobs = -1
        self.method = method
        self.kwargs = kwargs
        self.X_resampled = None
        self.y_resampled = None
        if self.method == 'over':
            from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
            if 'sampling_strategy' in self.kwargs:
                self.kwargs['sampling_strategy'] = self.kwargs['sampling_strategy']
            self.sampler = RandomOverSampler(**self.kwargs)
        elif self.method == 'under':
            from imblearn.under_sampling import RandomUnderSampler
            if 'sampling_strategy' in self.kwargs:
                self.kwargs['sampling_strategy'] = self.kwargs['sampling_strategy']
            self.sampler = RandomUnderSampler(**self.kwargs)
        elif self.method == 'smote':
            from imblearn.over_sampling import SMOTE
            self.sampler = SMOTE(**self.kwargs)
        elif self.method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            self.sampler = ADASYN(**self.kwargs)
        elif self.method == 'borderlinesmote':
            from imblearn.over_sampling import BorderlineSMOTE
            self.sampler = BorderlineSMOTE(**self.kwargs)
        else:
            raise NotImplementedError

    def fit(self, X, y):
        X = X.to_frame() if isinstance(X, pd.Series) else X
        y = y.to_frame() if isinstance(y, pd.Series) else y
        self.X_resampled, self.y_resampled = self.sampler.fit_resample(X, y)
        self.X_resampled = self.X_resampled.to_frame() if isinstance(self.X_resampled, pd.Series) else self.X_resampled
        self.y_resampled = self.y_resampled.to_frame() if isinstance(self.y_resampled, pd.Series) else self.y_resampled
        self.X_resampled, self.y_resampled = self.X_resampled.iloc[len(X):, :], self.y_resampled.iloc[len(y):, :]
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X
        y = y.to_frame() if isinstance(y, pd.Series) else y
        assert (X.columns == self.X_resampled.columns).all() & (y.columns == self.y_resampled.columns).all()
        X_all = pd.concat([X, self.X_resampled], axis=0)
        y_all = pd.concat([y, self.y_resampled], axis=0)
        return X_all, y_all

    @property
    def new_sample(self):
        return self.X_resampled, self.y_resampled

    def clear(self):
        self.__cache = None
        self.X_resampled = None
        self.y_resampled = None
        gc.collect()
