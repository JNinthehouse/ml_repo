import numpy as np
import torch


class focal_loss:
    def __init__(self, gamma=1.2, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def _binary_focal_loss(self, y_true, y_pred):
        y_pred = torch.from_numpy(y_pred).to(dtype=torch.float64)
        y_pred = torch.sigmoid(y_pred)
        y_pred.requires_grad = True
        y_true = torch.from_numpy(y_true).to(dtype=torch.float64)
        loss = torch.mean(-self.alpha * y_true * torch.pow(1 - y_pred, self.gamma) * torch.log(y_pred)
                          -
                          (1 - self.alpha) * (1 - y_true) * torch.pow(y_pred, self.gamma) * torch.log(1 - y_pred))
        grad1 = torch.autograd.grad(loss, y_pred, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1.sum(), y_pred)[0]
        grad1 = grad1.detach().numpy()
        grad2 = grad2.detach().numpy()
        return grad1, grad2

    def __call__(self, *args, **kwargs):
        return self._binary_focal_loss(*args, **kwargs)


class smooth_MSE:
    def __init__(self, delta=1.0):
        self.delta = delta

    def _smooth_MSE(self, y_true, y_pred):
        y_pred = torch.from_numpy(y_pred).to(dtype=torch.float64)
        y_pred.requires_grad = True
        y_true = torch.from_numpy(y_true).to(dtype=torch.float64)
        absolute_errors = torch.abs(y_true - y_pred)
        huber_loss = torch.where(absolute_errors < self.delta,
                                 0.5 * absolute_errors ** 2,
                                 self.delta * (absolute_errors - 0.5 * self.delta))
        loss = torch.mean(huber_loss)
        grad1 = torch.autograd.grad(loss, y_pred, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1.sum(), y_pred)[0]
        grad1 = grad1.detach().numpy()
        grad2 = grad2.detach().numpy()
        return grad1, grad2

    def __call__(self, *args, **kwargs):
        return self._smooth_MSE(*args, **kwargs)


if __name__ == '__main__':
    import numpy as np
    import lightgbm as lgb
    import sklearn.model_selection as ms
    import sklearn.metrics as mc
    import sklearn.neural_network as nn
    from feature_engineering.data_process import Scaler, Normalizer
    from feature_engineering.Pipe import PipeLine
    import pandas as pd
    import matplotlib.pyplot as plt

    x = np.random.randn(1000, 15)
    w = 4 * np.random.randn(x.shape[1], 1).round(1)
    bias = 0.1 * np.random.randn(x.shape[0]).reshape(-1, 1)
    y = np.dot(x, w) + bias
    x, y = pd.DataFrame(x), pd.DataFrame(y)
    y = y.applymap(lambda x: 1 if x > 0 else 0)


    def templacte(x, y):
        def scorer(model, x, y):
            if hasattr(model, 'objective'):
                if model.objective.__class__.__name__ == 'focal_loss':
                    y_pred = model.predict(x)
                    y_proba = 1 / (1 + np.exp(-y_pred))
                    y_pred = np.where(y_proba > 0.5, 1, 0)
                else:
                    y_pred = model.predict(x).ravel()
                    y_proba = model.predict_proba(x)[:, 1]
            else:
                y_pred = model.predict(x).ravel()
                y_proba = model.predict_proba(x)[:, 1]
            # y = process.inverse_transform(y)
            y = y.values.ravel()
            res = {}
            res['accuracy'] = mc.accuracy_score(y, y_pred)
            res['precision'] = mc.precision_score(y, y_pred)
            res['recall'] = mc.recall_score(y, y_pred)
            res['auc'] = mc.roc_auc_score(y, y_proba)
            res['balanced_accuracy'] = mc.balanced_accuracy_score(y, y_pred)
            return res

        x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.3, random_state=0)
        model1 = lgb.LGBMClassifier()
        model2 = lgb.LGBMClassifier()
        params = {
            'boosting_type': 'gbdt',
            'num_leaves': 25,
            'max_depth': 6,
            'n_estimators': 1000,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'min_child_samples': 10,
        }
        model1 = model1.set_params(**params)
        model2 = model2.set_params(**params)
        model2 = model2.set_params(objective=focal_loss(alpha=0.5, gamma=1.7))
        model3 = nn.MLPClassifier(hidden_layer_sizes=(128, 15), learning_rate='adaptive', max_iter=1000,
                                  early_stopping=True,
                                  n_iter_no_change=10)

        # process = PipeLine([Scaler()])
        # y_train = process.fit_transform(y_train)
        model1.fit(x_train, y_train)
        model2.fit(x_train, y_train)
        model3.fit(x_train, y_train)

        print(scorer(model1, x_test, y_test))
        print(scorer(model2, x_test, y_test))
        print(scorer(model3, x_test, y_test))

        # fig,axs = plt.subplots(1,3,figsize=(15,10))
        # fig.show()


    templacte(x, y)
