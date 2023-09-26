from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from scipy.interpolate import make_interp_spline, lagrange
import sklearn.gaussian_process as gp


class Bspline(BaseEstimator, TransformerMixin):
    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        X = np.atleast_1d(X)
        y = np.asarray(y).ravel()
        self.__model = make_interp_spline(X, y, k=self.k)
        return self

    def predict(self, X):
        X = np.asarray(X)
        X = np.atleast_1d(X)
        return self.__model(X)

    def transform(self, X):
        return self.predict(X)


class Lagrange(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        X = np.atleast_1d(X)
        y = np.asarray(y).ravel()
        self.__model = lagrange(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        X = np.atleast_1d(X)
        return self.__model(X)

    def transform(self, X):
        return self.predict(X)


class Poly(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        X = np.atleast_1d(X)
        y = np.asarray(y).ravel()
        self.coef_ = np.polyfit(X, y, self.degree)
        self.__model = np.poly1d(self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X).ravel()
        X = np.atleast_1d(X)
        return self.__model(X)

    def transform(self, X):
        return self.predict(X)


class GPRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, kernel=None, n_restarts_optimizer=5):
        self.kernel = gp.kernels.ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * gp.kernels.RBF(
            length_scale=0.5, length_scale_bounds=(1e-4, 1e4)) if kernel is None else kernel
        self.__model = gp.GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts_optimizer)

    def fit(self, X, y=None):
        X = np.asarray(X)
        X = np.atleast_1d(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)
        self.__model.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X)
        X = np.atleast_1d(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if return_std:
            return self.__model.predict(X, return_std=True)
        else:
            return self.__model.predict(X)

    def transform(self, X):
        return self.predict(X)
