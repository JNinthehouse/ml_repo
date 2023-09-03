import numpy as np
import sklearn.datasets as ds
import pandas as pd
from feature_engineering.Pipe import PipeLine
from feature_engineering.data_process import OverSampler, TypeTransfer
import sklearn.model_selection as ms

def get_data(n_samples=100000):
    generator = ds.make_classification
    X, y = generator(n_samples=n_samples,
                     n_features=18,
                     n_informative=10,
                     n_redundant=4,
                     n_repeated=0,
                     n_classes=2,
                     class_sep=0.8,
                     weights={0: 0.9, 1: 0.1},
                     flip_y=0.1)
    return pd.DataFrame(X, columns=[f'f{i + 1}' for i in range(X.shape[1])]), pd.DataFrame(y, columns=['target'])


def lgb_base(random_state=None):
    import lightgbm as lgb
    params_lgb = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 25,
        'max_depth': 6,
        'subsample': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': None,
        'n_jobs': -1,
        'min_child_samples': 20,
    }
    sampler = OverSampler(method='random', random_state=random_state, sampling_strategy={1: 2.0})
    transfer = TypeTransfer(to_type='dataframe', verbose=False)
    model = lgb.LGBMClassifier(**params_lgb)
    return PipeLine([transfer, sampler, model])



def scorer(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    import sklearn.metrics as mc
    import pandas as pd
    if len(np.unique(y)) > 1:
        auc = mc.roc_auc_score(y, y_proba)
    else:
        auc = None
    res = {
        'auc': auc,
        'accuracy': mc.accuracy_score(y, y_pred),
        'precision': mc.precision_score(y, y_pred),
        'recall': mc.recall_score(y, y_pred),
        'f1': mc.f1_score(y, y_pred),
    }
    return pd.Series(res)
