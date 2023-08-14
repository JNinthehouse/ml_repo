class PipeLine:
    def __init__(self,steps,verbose=False):
        from sklearn.pipeline import make_pipeline
        self.steps=steps
        self.verbose=verbose
        if not isinstance(self.steps,list | tuple):
            if hasattr(self.steps,'fit') and hasattr(self.steps,'transform') and hasattr(self.steps,'fit_transform'):
                self.steps=[self.steps]
            else:
                raise Exception(f'The steps must be the list or tuple or the object which has the fit and transform method')
        else:
            for step in self.steps:
                if not (hasattr(step,'fit') and hasattr(step,'transform') and hasattr(step,'fit_transform')):
                    raise Exception(f'The steps must be the list or tuple or the object which has the fit and transform method')
                else:
                    continue
        self.pipeline = make_pipeline(*self.steps)

    def __repr__(self):
        return self.pipeline.__repr__()

    def __add__(self, other):
        from sklearn.pipeline import Pipeline
        if isinstance(other,PipeLine):
            steps=self.steps+other.steps
            return PipeLine(steps=steps, verbose=self.verbose)
        elif isinstance(other,Pipeline):
            steps=self.steps+[step for _,step in other.steps]
            return PipeLine(steps=steps, verbose=self.verbose)
        else:
            raise Exception(f'The other must be the PipeLine')

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, item):
        return PipeLine(steps=self.steps[item],verbose=self.verbose)

    def __iter__(self):
        return iter(self.steps)

    def __next__(self):
        return next(self.steps)

    def fit(self,X,y=None,**kwargs):
        if len(self.steps) == 1:
            return self.pipeline.fit(X,y,**kwargs)
        else:
            X_res, y_res = X, y
            for i in range(len(self.steps)-1):
                res = self.steps[i].fit_transform(X_res,y_res,**kwargs)
                if isinstance(res,tuple):
                    X_res, y_res = res
                else:
                    X_res = res
            self.steps[-1].fit(X_res,y_res,**kwargs)
            return self.pipeline

    def transform(self,X,y=None,**kwargs):
        if len(self.steps) == 1:
            return self.pipeline.transform(X,y,**kwargs)
        else:
            X_res, y_res = X, y
            for i in range(len(self.steps)-1):
                res = self.steps[i].transform(X_res,y_res,**kwargs)
                if isinstance(res,tuple):
                    X_res, y_res = res
                else:
                    X_res = res
            return self.steps[-1].transform(X_res,y_res,**kwargs)

    def fit_transform(self,X,y=None,**fit_params):
        self.fit(X,y,**fit_params)
        return self.transform(X,y,**fit_params)


    def predict(self,X,y=None,**kwargs):
        return self.pipeline.predict(X,y,**kwargs)

    def predict_proba(self,X,y=None,**kwargs):
        return self.pipeline.predict_proba(X,y,**kwargs)

    def predict_log_proba(self,X,y=None,**kwargs):
        return self.pipeline.predict_log_proba(X,y,**kwargs)

    def score(self,X,y=None,**kwargs):
        return self.pipeline.score(X,y,**kwargs)

    def inverse_transform(self,X,y=None,**kwargs):
        return self.pipeline.inverse_transform(X,y,**kwargs)

    def get_params(self,deep=True):
        return self.pipeline.get_params(deep=deep)

    def set_params(self,**params):
        return self.pipeline.set_params(**params)