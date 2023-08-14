from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd
from sklearn import model_selection as ms

if __name__ == '__main__':
    data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
    path = f'{data_url}train.csv'
    train_data = pd.read_csv(path)
    train_data = TabularDataset(train_data)
    X, y = train_data.iloc[:, :-1], train_data.iloc[:, -1]

    model = TabularPredictor(label=y.name,
                             problem_type='multiclass',
                             sample_weight='balance_weight')
    hyperparams={'GBM': [{'extra_trees': True,'ag_args': {'name_suffix': 'XT'}}, {}, 'GBMLarge']}
    model.fit(train_data=train_data,
              hyperparameters=hyperparams,
              presets='best_quality',
              num_bag_folds=5,
              num_bag_sets=2,
              num_stack_levels=1,
              )
