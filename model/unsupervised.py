import sklearn.cluster as cl
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import sklearn.metrics as mc
import hdbscan
import sklearn_extra.cluster as scl
import sys

sys.path.append(r'E:\ml_repo\model')
from denmune import DenMune


class KMedoidsCluster(scl.KMedoids, ClusterMixin, TransformerMixin):
    def fit(self, X, y=None):
        super().fit(X, y)
        sihouette_score = mc.silhouette_score(X, self.labels_, metric=self.metric)
        self.scores_ = sihouette_score
        return self


class SpecCluster(cl.SpectralClustering, ClusterMixin, TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        super().fit(X, y)
        sihouette_score = mc.silhouette_score(X, self.labels_, metric=self.affinity)
        self.scores_ = sihouette_score
        return self


class AggCluster(cl.AgglomerativeClustering, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters=2, compute_distances=True, **kwargs):
        super().__init__(n_clusters=n_clusters,
                         compute_distances=compute_distances,
                         **kwargs)

    def fit(self, X, y=None):
        if isinstance(X, pd.Series | pd.DataFrame):
            self.instance_index = X.index.tolist()
        else:
            self.instance_index = list(range(len(X)))
        super().fit(X, y)
        sihouette_score = mc.silhouette_score(X, self.labels_, metric=self._metric)
        self.scores_ = sihouette_score
        return self

    def cluster_graph(self, ax=None,
                      truncate_mode=None,
                      p=5,
                      orientation='top',
                      figsize=(20, 10)):
        assert truncate_mode in ['level', 'lastp', None]
        if self.compute_distances == False:
            raise Exception('Please set compute_distances to True and refit the model')

        counts = np.zeros(self.children_.shape[0])
        '''
        model.children_是一个n_sample-1*2的数组
        从每个样本单独为一个簇，到所有样本是一个簇，一共需要n_sample-1轮
        model.children_每一行([a,b])表示第i轮迭代，会将簇a和簇b合并
        如果簇a或簇b的编号比n_sample小，那么簇a或簇b表示单个样本组成的叶子节点
        如果簇a或簇b的编号比n_sample大，那么簇a或簇b表示第a-n_sample或第b-n_sample次合并后形成的新簇（非叶节点）
        count[i]需要计算的是，第i轮合并后，形成的新簇的大小
        '''
        n_samples = len(self.labels_)
        for i, merge in enumerate(self.children_):
            # merge是当前轮需要merge的两个节点的index组成的list
            # i的范围是0~n_samples-1
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                    # 其中一个合并的簇是编号为child_idx的叶子节点，所以当前合并的样本数量仅仅+1
                else:
                    current_count += counts[child_idx - n_samples]
                    # 其中一个合并的簇是第child_idx - n_samples轮合并后生成的新簇，所以当前合并的样本数量+第child_idx - n_samples轮合并后生成的新簇的大小
            counts[i] = current_count
        linkage_matrix = np.column_stack(
            [self.children_, self.distances_, counts]
        ).astype(float)

        from scipy.cluster.hierarchy import dendrogram
        if ax is None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=figsize)
            dendrogram(linkage_matrix,
                       truncate_mode=truncate_mode,
                       p=p,
                       labels=self.instance_index,
                       orientation=orientation)
        else:
            dendrogram(linkage_matrix,
                       ax=ax,
                       truncate_mode=truncate_mode,
                       p=p,
                       labels=self.instance_index,
                       orientation=orientation)
            return ax


class HDBSCAN(hdbscan.HDBSCAN, ClusterMixin, TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        X = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        y = y.to_frame() if isinstance(y, pd.Series) else (y.copy() if y is not None else None)
        super().fit(X, y)
        sihouette_score = mc.silhouette_score(X, self.labels_, metric=self.metric)
        self.scores_ = sihouette_score
        return self


class DistMetric:
    def __init__(self, weights):
        self.weights = weights

    def weighted_euclidean(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2 * self.weights))


class DenmuneCluster(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, k_neighbors=10):
        self.k_neighbors = k_neighbors
        self.embeddata = None
        self.__denmune = None

    def fit(self, X, y=None):
        self.__denmune = DenMune(train_data=X, k_nearest=self.k_neighbors)
        self.labels = self.__denmune.fit_predict(show_plots=False, validate=False)[0]['train']
        self.labels = pd.Series(self.labels)
        valid_labels_size = self.labels[self.labels >= 0].nunique()
        valid_mapper = {i: j for i, j in zip(self.labels.unique(), range(valid_labels_size))}
        noise_mapper = {i: -1 for i in self.labels.unique() if i not in valid_mapper.keys()}
        mapper = {**valid_mapper, **noise_mapper}
        self.labels = self.labels.map(mapper)
        self.labels = self.labels.values
        return self

    @property
    def labels_(self):
        return self.labels

    def plot(self, show_noise=False, ax=None):
        if self.__denmune is None:
            raise Exception('Please fit the model first')
        import matplotlib.pyplot as plt
        t0 = [i[0] for i in self.__denmune.data]
        t1 = [i[1] for i in self.__denmune.data]
        plot_data = pd.DataFrame({'t0': t0, 't1': t1, 'labels': self.labels})
        plot_data = plot_data[plot_data.labels >= 0] if not show_noise else plot_data
        if ax is None:
            plt.figure(figsize=(10, 10))
            plt.scatter(plot_data.t0, plot_data.t1, c=plot_data.labels, s=10)
            plt.show()
        else:
            ax.scatter(plot_data.t0, plot_data.t1, c=plot_data.labels, s=10)
            return ax
