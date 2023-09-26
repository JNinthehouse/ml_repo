from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cvxpy as cp


class WeightLearner(BaseEstimator, TransformerMixin):
    def __init__(self, loss):
        self.loss = loss

    def fit(self, X, y):
        import cvxpy as cp
        x = np.atleast_2d(np.asarray(X))
        y = np.asarray(y).ravel()
        n_features = x.shape[1]
        w = cp.Variable(n_features)
        objective = cp.Minimize(self.loss(y, x @ w))
        constraints = [0 <= w, w <= 1, cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='SCS')
        self.weight = w.value
        return self

    def predict(self, X):
        x = np.asarray(X)
        x = np.atleast_2d(x)
        return x @ self.weight.reshape(-1, 1)

    @property
    def coef_(self):
        return self.weight
