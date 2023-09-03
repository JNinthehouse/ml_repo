from typing import Callable, Any
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion, make_union

class PipeLine(Pipeline):
    def __init__(self, steps=None, memory=None, verbose=False):
        """

        :rtype: object
        """
        if steps is not None:
            if (isinstance(steps, list) and
                    all([isinstance(i, tuple) for i in steps]) and
                    all([len(i) == 2 for i in steps])):
                pass
            else:
                steps = make_pipeline(*steps).steps
        super().__init__(steps=steps, memory=memory, verbose=verbose)
        self.maskcode = {i[0]: True for i in self.steps}

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, ind):
        self_steps = [i[1] for i in self.steps]
        if isinstance(ind, slice):
            return PipeLine(steps=self_steps[ind])
        else:
            return self.steps[ind][1]

    def __add__(self, other):
        self_steps = [i[1] for i in self.steps]
        other_steps = [i[1] for i in other.steps]
        return PipeLine(steps=self_steps + other_steps)

    def append(self, step):
        self_steps = [i[1] for i in self.steps]
        self_steps.append(step)
        return PipeLine(steps=self_steps)

    def fit(self, X, y=None, **fit_params):
        if len(self.steps) == 1:
            if self.maskcode[list(self.maskcode.keys())[0]]:
                return self.steps[0][1].fit(X, y, **fit_params)
        else:
            for name, steps in self.steps[:-1]:
                if self.maskcode[name]:
                    res = steps.fit_transform(X, y, **fit_params)
                    if y is None:
                        X = res
                    else:
                        X, y = res
            if self.maskcode[list(self.maskcode.keys())[-1]]:
                return self.steps[-1][1].fit(X, y, **fit_params)

    def transform(self, X, y=None, **fit_params):
        if len(self.steps) == 1:
            if self.maskcode[list(self.maskcode.keys())[0]]:
                return self.steps[0][1].transform(X, y, **fit_params)
            else:
                return (X, y) if y is not None else X
        else:
            for name, steps in self.steps:
                if self.maskcode[name]:
                    res = steps.transform(X, y, **fit_params)
                    if y is None:
                        X = res
                    else:
                        X, y = res
            return (X, y) if y is not None else X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)

    def mask(self, maskcode=None):
        from collections import OrderedDict
        if maskcode is None:
            return self
        else:
            if isinstance(maskcode, str):
                self.maskcode[maskcode] = False
            elif isinstance(maskcode, int):
                self.maskcode[list(self.maskcode.keys())[maskcode]] = False
            elif isinstance(maskcode, list | tuple):
                assert len(maskcode) <= len(self.steps), 'maskcode must be less than or equal to steps'
                if len(maskcode) == len(self.steps):
                    if (all([isinstance(i, bool) for i in maskcode])):
                        self.maskcode = OrderedDict({i[0]: j for i, j in zip(self.steps, maskcode)})
                    elif (all([(i == 1) or (i == 0) for i in maskcode])):
                        self.maskcode = OrderedDict({i[0]: bool(j) for i, j in zip(self.steps, maskcode)})
                    else:
                        raise ValueError('maskcode must be bool or 0/1')
                else:
                    assert len(np.unique(maskcode)) == len(maskcode), 'maskcode must be unique'
                    if all([isinstance(i, str) for i in maskcode]):
                        self.maskcode = OrderedDict({i[0]: (i[0] not in maskcode) for i in self.steps})
                    elif all([isinstance(i, int) for i in maskcode]):
                        assert max(maskcode) in range(len(self.steps)), 'maskcode must be less than or equal to steps'
                        self.maskcode = OrderedDict({i[0]: (j not in maskcode) for j, i in enumerate(self.steps)})
                    else:
                        raise ValueError('maskcode must be str or int')
            else:
                raise ValueError('maskcode must be str or int or list or tuple')

    def unmask(self, maskcode=None):
        from collections import OrderedDict
        if maskcode is None:
            return self
        else:
            if isinstance(maskcode, str):
                self.maskcode[maskcode] = True
            elif isinstance(maskcode, int):
                self.maskcode[list(self.maskcode.keys())[maskcode]] = True
            elif isinstance(maskcode, list | tuple):
                assert len(maskcode) <= len(self.steps), 'maskcode must be less than or equal to steps'
                if len(maskcode) == len(self.steps):
                    if (all([isinstance(i, bool) for i in maskcode])):
                        self.maskcode = OrderedDict({i[0]: j for i, j in zip(self.steps, maskcode)})
                    elif (all([(i == 1) or (i == 0) for i in maskcode])):
                        self.maskcode = OrderedDict({i[0]: bool(j) for i, j in zip(self.steps, maskcode)})
                    else:
                        raise ValueError('maskcode must be bool or 0/1')
                else:
                    assert len(np.unique(maskcode)) == len(maskcode), 'maskcode must be unique'
                    if all([isinstance(i, str) for i in maskcode]):
                        for i in maskcode:
                            self.maskcode[i] = True
                    elif all([isinstance(i, int) for i in maskcode]):
                        assert max(maskcode) in range(len(self.steps)), 'maskcode must be less than or equal to steps'
                        for i in maskcode:
                            self.maskcode[list(self.maskcode.keys())[i]] = True
                    else:
                        raise ValueError('maskcode must be str or int')
            else:
                raise ValueError('maskcode must be str or int or list or tuple')

    def get_params(self, deep=True):
        return super().get_params(deep=deep)

    def set_params(self, **params):
        return super().set_params(**params)

    def _data_flow(self, X, y=None, **params):
        if len(self.steps) == 1:
            return (X, y) if y is not None else X
        else:
            for name, steps in self.steps[:-1]:
                if self.maskcode[name]:
                    res = steps.fit_transform(X, y, **params)
                    if y is None:
                        X = res
                    else:
                        X, y = res
            return (X, y) if y is not None else X

    def predict(self, X, **predict_params):
        X = self._data_flow(X, **predict_params)
        if self.maskcode[list(self.maskcode.keys())[-1]] and hasattr(self.steps[-1][1], 'predict'):
            return self.steps[-1][1].predict(X, **predict_params)

    def predict_proba(self, X, **predict_params):
        X = self._data_flow(X, **predict_params)
        if self.maskcode[list(self.maskcode.keys())[-1]] and hasattr(self.steps[-1][1], 'predict_proba'):
            return self.steps[-1][1].predict_proba(X, **predict_params)

    def predict_log_proba(self, X, **predict_params):
        X = self._data_flow(X, **predict_params)
        if self.maskcode[list(self.maskcode.keys())[-1]] and hasattr(self.steps[-1][1], 'predict_log_proba'):
            return self.steps[-1][1].predict_log_proba(X, **predict_params)

    def score(self, X, y=None, **predict_params):
        X = self._data_flow(X, **predict_params)
        if self.maskcode[list(self.maskcode.keys())[-1]] and hasattr(self.steps[-1][1], 'score'):
            return self.steps[-1][1].score(X, y, **predict_params)

    def decision_function(self, X, **predict_params):
        X = self._data_flow(X, **predict_params)
        if self.maskcode[list(self.maskcode.keys())[-1]] and hasattr(self.steps[-1][1], 'decision_function'):
            return self.steps[-1][1].decision_function(X, **predict_params)

    def inverse_transform(self, X, y=None, **fit_params):
        if len(self.steps) == 1:
            if self.maskcode[list(self.maskcode.keys())[0]]:
                return self.steps[0][1].inverse_transform(X, y, **fit_params)
            else:
                return (X, y) if y is not None else X
        else:
            for name, steps in self.steps[::-1]:
                if self.maskcode[name]:
                    res = steps.inverse_transform(X, y, **fit_params)
                    if y is None:
                        X = res
                    else:
                        X, y = res
            return (X, y) if y is not None else X


class PipeUnion(FeatureUnion):
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        if transformer_list is not None:
            if (isinstance(transformer_list, list) and
                    all([isinstance(i, tuple) for i in transformer_list]) and
                    all([len(i) == 2 for i in transformer_list])):
                pass
            else:
                transformer_list = make_union(*transformer_list,
                                              n_jobs=n_jobs, verbose=verbose).transformer_list
        super().__init__(transformer_list=transformer_list, n_jobs=n_jobs,
                         verbose=verbose, transformer_weights=transformer_weights)

    def fit(self, X, y=None, **fit_params):
        from sklearn.utils.parallel import Parallel, delayed
        from multiprocessing import cpu_count
        match self.n_jobs:
            case None:
                n_jobs = 1
            case -1:
                n_jobs = cpu_count()
            case int:
                n_jobs = self.n_jobs

        def _func_tmp(trans, X, y, **fit_params):
            trans.fit(X, y, **fit_params)

        func_tmp = lambda trans: _func_tmp(trans, X, y, **fit_params)
        Parallel(n_jobs=n_jobs)(delayed(func_tmp)(trans[1]) for trans in self.transformer_list)
        return self

    def _transform(self, X, y=None, transform_type='transform', **fit_params):
        from sklearn.utils.parallel import Parallel, delayed
        from multiprocessing import cpu_count
        match self.n_jobs:
            case None:
                n_jobs = 1
            case -1:
                n_jobs = cpu_count()
            case int:
                n_jobs = self.n_jobs

        transform_type = fit_params.get('transform_type', 'transform')

        def _func_tmp(trans, X, y, **fit_params):
            y = y.to_frame() if isinstance(y, pd.Series) else y
            if transform_type == 'transform':
                res = trans.transform(X, y, **fit_params)
            elif transform_type == 'inverse_transform':
                res = trans.inverse_transform(X, y, **fit_params)
            else:
                raise ValueError('transform_type must be transform or inverse_transform')
            if isinstance(res, tuple | list) and len(res) == 2:
                X_res, y_res = res
                y_res = y_res.to_frame() if isinstance(y_res, pd.Series) else y_res
                if (y_res == y).all().iloc[0] == True:
                    ind = 1
                else:
                    ind = 2
            else:
                X_res, y_res, ind = res, None, 0
            return X_res, y_res, ind

        func_tmp: Callable[[Any], Any] = lambda trans: _func_tmp(trans, X, y, **fit_params)
        res_list = Parallel(n_jobs=n_jobs)(delayed(func_tmp)(trans[1]) for trans in self.transformer_list)

        X_res = pd.concat([i[0] for i in res_list], axis=1)
        y_res = None
        ind = np.array([i[2] for i in res_list])
        ind_2 = np.where(ind == 2)[0]
        if len(ind_2) == 1:
            y_res = res_list[ind_2[0]][1]
        elif len(ind_2) > 1:
            y_res = pd.concat([res_list[i][1] for i in ind_2], axis=1)
            if y_res.apply(lambda x: x.nunique(), axis=1).sum() == len(y_res):
                y_res = y_res.iloc[:, 0]
            else:
                raise ValueError('Transformed y must be unique, otherwise it will be ambiguous')
        else:
            ind_1 = np.where(ind == 1)[0]
            if len(ind_1) > 0:
                y_res = res_list[ind_1[0]][1]
            else:
                pass
        return (X_res, y_res) if y_res is not None else X_res

    def transform(self, X, y=None, **fit_params):
        return self._transform(X, y, transform_type='transform')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)

    def inverse_transform(self, X, y=None, **fit_params):
        return self._transform(X, y, transform_type='inverse_transform')

    def predict(self, X, **predict_params):
        from sklearn.utils.parallel import Parallel, delayed
        from multiprocessing import cpu_count
        match self.n_jobs:
            case None:
                n_jobs = 1
            case -1:
                n_jobs = cpu_count()
            case int:
                n_jobs = self.n_jobs
        X = X.to_frame() if isinstance(X, pd.Series) else X

        def _func_tmp(trans, X, **predict_params):
            try:
                res = trans.predict(X, **predict_params)
            except:
                res = None
            return res

        func_tmp = lambda trans: _func_tmp(trans, X, **predict_params)
        res_list = Parallel(n_jobs=n_jobs)(delayed(func_tmp)(trans[1]) for trans in self.transformer_list)
        return tuple(res_list)

    def predict_proba(self, X, **predict_params):
        from sklearn.utils.parallel import Parallel, delayed
        from multiprocessing import cpu_count
        match self.n_jobs:
            case None:
                n_jobs = 1
            case -1:
                n_jobs = cpu_count()
            case int:
                n_jobs = self.n_jobs
        X = X.to_frame() if isinstance(X, pd.Series) else X

        def _func_tmp(trans, X, **predict_params):
            try:
                res = trans.predict_proba(X, **predict_params)
            except:
                res = None
            return res

        func_tmp = lambda trans: _func_tmp(trans, X, **predict_params)
        res_list = Parallel(n_jobs=n_jobs)(delayed(func_tmp)(trans[1]) for trans in self.transformer_list)
        return tuple(res_list)

    def append(self, Pipeline):
        if isinstance(Pipeline, PipeLine):
            lists = [i[1] for i in self.transformer_list] + [Pipeline]
            self.transformer_list = make_union(*lists).transformer_list
        else:
            raise ValueError('the input must be PipeLine')

    def extend(self, Pipeunion):
        if isinstance(Pipeunion, PipeUnion):
            lists = [i[1] for i in self.transformer_list] + [i[1] for i in Pipeunion.transformer_list]
            self.transformer_list = make_union(*lists).transformer_list
        else:
            raise ValueError('the input must be PipeUnion')


if __name__ == '__main__':
    from feature_engineering.data_process import Scaler

    pipe = PipeLine([Scaler()])
