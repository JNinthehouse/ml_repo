import sklearn.cluster as cl
import pandas as pd


class KMeans(cl.KMeans):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        res = self.transform(X)
        return pd.DataFrame(res, columns=[f'cluster_{i}' for i in range(res.shape[1])], index=X.index)


class SpectralClustering(cl.SpectralClustering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        res = self.transform(X)
        return pd.DataFrame(res, columns=[f'cluster_{i}' for i in range(res.shape[1])], index=X.index)


class AgglomerativeClustering(cl.AgglomerativeClustering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        res = self.transform(X)
        return pd.DataFrame(res, columns=[f'cluster_{i}' for i in range(res.shape[1])], index=X.index)
