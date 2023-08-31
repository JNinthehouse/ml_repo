import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline


class PipeLine(Pipeline):
    def __init__(self, steps=None, memory=None, verbose=False) -> object:
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
        return PipeLine(**params)

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
