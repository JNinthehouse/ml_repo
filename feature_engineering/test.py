import sklearn.datasets as ds
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import sklearn.ensemble as es
import pandas as pd
from feature_engineering.feature_extraction import FeatureBoostingGenerator
from feature_engineering.data_process import Normalizer

if __name__ == '__main__':
    data = ds.load_breast_cancer(as_frame=True)
    df = data['data']
    df['target'] = data['target']
    X, y = df.iloc[:, :-1], df.iloc[:, -1]


    def test_score(X, y):

        params = {
            'objective': 'binary',
            'importance_type': 'gain',
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': 500,
            'num_leaves': 50,
            'max_depth': 10,
            'learning_rate': 0.15,
        }
        lgb_base = lgb.LGBMClassifier(**params, verbose=-1)
        lgb_bag = es.BaggingClassifier(estimator=lgb_base, n_estimators=10, random_state=10)
        lgb_weighted = es.VotingClassifier(estimators=[('lgb', lgb_base), ('lgb_bag', lgb_bag)],
                                           voting='soft')
        fbg = FeatureBoostingGenerator(numerical_features=X.columns.to_list()[:3])
        X, y = fbg.fit_transform(X, y)
        pt = Normalizer()
        X, y = pt.fit_transform(X, y)

        def get_score(model, X_test, y_test):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            score = pd.DataFrame({'f1': [f1_score(y_test, y_pred)],
                                  'auc': [roc_auc_score(y_test, y_proba[:, 1])],
                                  'accuracy': [accuracy_score(y_test, y_pred)]})
            return score

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
        score_base = pd.DataFrame(columns=['f1', 'auc', 'accuracy'])
        score_bag = pd.DataFrame(columns=['f1', 'auc', 'accuracy'])
        score_weighted = pd.DataFrame(columns=['f1', 'auc', 'accuracy'])
        for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
            print(f'Fold {i + 1}')
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            lgb_bag.fit(X_train, y_train)
            lgb_base.fit(X_train, y_train)
            lgb_weighted.fit(X_train, y_train)
            score_base = pd.concat([score_base, get_score(lgb_base, X_test, y_test)], axis=0)
            score_bag = pd.concat([score_bag, get_score(lgb_bag, X_test, y_test)], axis=0)
            score_weighted = pd.concat([score_weighted, get_score(lgb_weighted, X_test, y_test)], axis=0)
        score_base = score_base.mean(axis=0).rename('base')
        score_bag = score_bag.mean(axis=0).rename('bag')
        score_weighted = score_weighted.mean(axis=0).rename('weighted')
        score = pd.concat([score_base, score_bag, score_weighted], axis=1)
        return score
