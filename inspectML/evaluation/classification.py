import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def SIL(X, labels):
    """Silhouette Score"""
    X, labels = np.array(X), np.array(labels)
    return silhouette_score(X, labels)

def DBI(X, labels):
    """Davies-Bouldin Index"""
    X, labels = np.array(X), np.array(labels)
    return davies_bouldin_score(X, labels)

def CHI(X, labels):
    """Calinski-Harabasz Index"""
    X, labels = np.array(X), np.array(labels)
    return calinski_harabasz_score(X, labels)

def DI(X, labels):
    """Dunn Index"""
    X, labels = np.array(X), np.array(labels)
    distances = pairwise_distances(X)
    clusters = [np.where(labels == c)[0] for c in np.unique(labels)]

    def min_inter():
        return min(
            np.min(distances[np.ix_(c1, c2)])
            for i, c1 in enumerate(clusters)
            for c2 in clusters[i + 1 :]
        )

    def max_intra():
        return max(np.max(distances[np.ix_(c, c)]) for c in clusters)

    denom = max_intra()
    return np.inf if denom == 0 else min_inter() / denom

def inert(X, labels):
    """Inertia"""
    X, labels = np.array(X), np.array(labels)
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=0).fit(X)
    _, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
    return np.sum(distances ** 2)

def ARI(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return adjusted_rand_score(y_true, y_pred)

def F1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_binary = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_binary = label_binarize(y_pred, classes=np.unique(y_pred))
    return f1_score(y_true_binary, y_pred_binary, average="macro")

def NMI(y_true, y_pred):
    """Normalized Mutual Information"""
    from sklearn.metrics import normalized_mutual_info_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return normalized_mutual_info_score(y_true, y_pred)

def HOM(y_true, y_pred):
    """Homogeneity Score"""
    from sklearn.metrics import homogeneity_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return homogeneity_score(y_true, y_pred)

def COMP(y_true, y_pred):
    """Completeness Score"""
    from sklearn.metrics import completeness_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return completeness_score(y_true, y_pred)
def VMEAS(y_true, y_pred):
    """V-Measure"""
    from sklearn.metrics import v_measure_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return v_measure_score(y_true, y_pred)
