import sklearn.cluster as cl
import pandas as pd
import sklearn.metrics as mc
import hdbscan

class KMeans(cl.KMeans):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.silhouette_score = mc.silhouette_score(X, self.labels_, metric='euclidean')
        self.calinski_harabasz_score = mc.calinski_harabasz_score(X, self.labels_)
        self.davies_bouldin_score = mc.davies_bouldin_score(X, self.labels_)
        return self

    def transform(self, X):
        res = self.transform(X)
        return pd.DataFrame(res, columns=[f'cluster_{i}' for i in range(res.shape[1])], index=X.index)


class SpectralClustering(cl.SpectralClustering):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X):
        res = self.transform(X)
        return pd.DataFrame(res, columns=[f'cluster_{i}' for i in range(res.shape[1])], index=X.index)


class AgglomerativeClustering(cl.AgglomerativeClustering):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X):
        res = self.transform(X)
        return pd.DataFrame(res, columns=[f'cluster_{i}' for i in range(res.shape[1])], index=X.index)


class HDBSCAN(hdbscan.HDBSCAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X):
        res = self.transform(X)
        return pd.DataFrame(res, columns=[f'cluster_{i}' for i in range(res.shape[1])], index=X.index)


if __name__ == '__main__':
    import sklearn.datasets as ds

    data = ds.load_iris(as_frame=True)
    df = data['data']
    model = KMeans(n_clusters=5)
    model.fit(df)
    model.score(df)
    from sklearn_extra.robust import RobustWeightedClassifier
