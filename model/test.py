import sklearn.datasets as ds
import pandas as pd
from feature_engineering.PipeLine import PipeLine
from feature_engineering.data_process import UnderSampler, TypeTransfer
import sklearn.model_selection as ms

def get_data(n_samples=100000):
    generator = ds.make_classification
    X, y = generator(n_samples=n_samples,
                     n_features=18,
                     n_informative=10,
                     n_redundant=4,
                     n_repeated=0,
                     n_classes=2,
                     class_sep=1.8,
                     weights=[0.05, 0.95],
                     flip_y=0.1)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(18)]), pd.DataFrame(y, columns=['target'])


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
    sampler = UnderSampler(random_state=random_state, sampling_strategy={0: 0.1, 1: 0.9}, replacement=True)
    transfer = TypeTransfer(to_type='dataframe', verbose=False)
    model = lgb.LGBMClassifier(**params_lgb)
    return PipeLine([transfer, sampler, model])


def EnsembelModel(n_estimators=100):
    from sklearn.ensemble import VotingClassifier
    model_list = [('lgb' + str(i), lgb_base(random_state=i)) for i in range(n_estimators)]
    return VotingClassifier(estimators=model_list, voting='soft', n_jobs=-1)


def eval(VotingModel):
    for estimator in VotingModel.estimators_:
        estimator.mask('undersampler')


def train(VotingModel):
    for estimator in VotingModel.estimators_:
        maskcode = [True for _ in range(len(estimator))]
        estimator.unmask(maskcode)


def scorer(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    import sklearn.metrics as mc
    import pandas as pd
    res = {
        # 'auc': mc.roc_auc_score(y, y_proba),
        'accuracy': mc.accuracy_score(y, y_pred),
        'precision': mc.precision_score(y, y_pred),
        'recall': mc.recall_score(y, y_pred),
        'f1': mc.f1_score(y, y_pred),
    }
    return pd.Series(res)


def lr_ensemble():
    import sklearn.linear_model as sl
    import sklearn.ensemble as se
    lr_base = sl.LogisticRegression(solver='liblinear',
                                    max_iter=1000,
                                    n_jobs=-1)
    se.AdaBoostClassifier


def svm_base():
    import sklearn.svm as ss
    return ss.SVC(kernel='rbf',
                  C=1.1,
                  degree=3,
                  probability=True,
                  max_iter=1000)


def knn_base():
    import sklearn.neighbors as sn
    return sn.KNeighborsClassifier(n_neighbors=6,
                                   weights='distance',
                                   n_jobs=-1)




if __name__ == '__main__':
    x, y = get_data()
    x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=1)
    vmodel = EnsembelModel(n_estimators=10)
    train(vmodel)
    vmodel.fit(x_train, y_train)
    eval(vmodel)
    print(scorer(vmodel, x_test, y_test))
