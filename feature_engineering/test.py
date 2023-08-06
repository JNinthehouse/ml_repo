import sklearn.datasets as ds
import pandas as pd

if __name__ == '__main__':
    data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
    path = f'{data_url}train.csv'
    train_data = pd.read_csv(path)
    X, y = train_data.iloc[:, :-1], train_data.iloc[:, -1]
