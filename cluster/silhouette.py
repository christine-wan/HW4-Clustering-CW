import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        Initializes the Silhouette class. No input data needed initially.
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the silhouette score for each of the observations.

        Inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.
            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`.

        Outputs:
            np.ndarray
                A 1D array with the silhouette scores for each of the observations in X.
        """
        # Input validation
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data X or y cannot be empty.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X must match the number of labels in y.")

        unique_labels = np.unique(y)
        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)
        dist_mat = self._get_distance_matrix(X)

        for i in range(n_samples):
            current_label = y[i]

            # Compute a(i) - Mean intra-cluster distance
            a_i = self._get_intra_cluster_distance(i, dist_mat, y, current_label)

            # If a_i is NaN, set silhouette score to NaN
            if np.isnan(a_i):
                silhouette_scores[i] = np.nan
                continue

            # Compute b(i) - Mean nearest-cluster distance
            b_i = self._get_nearest_cluster_distance(i, dist_mat, y, unique_labels, current_label)

            # Compute silhouette score for the current sample
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return silhouette_scores

    def _get_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise distance matrix for the dataset.
        """
        return cdist(X, X, metric='euclidean')

    def _get_intra_cluster_distance(self, i: int, dist_mat: np.ndarray, y: np.ndarray, current_label: int) -> float:
        """
        Computes the mean intra-cluster distance for a given sample.
        """
        same_cluster_mask = y == current_label
        if np.sum(same_cluster_mask) > 1:
            return np.mean(dist_mat[i, same_cluster_mask][dist_mat[i, same_cluster_mask] > 0])
        return np.nan  # Return NaN if the cluster has only one point

    def _get_nearest_cluster_distance(self, i: int, dist_mat: np.ndarray, y: np.ndarray, unique_labels: np.ndarray,
                                      current_label: int) -> float:
        """
        Computes the mean distance to the nearest cluster for a given sample.
        """
        b_i = np.inf
        for label in unique_labels:
            if label == current_label:
                continue
            other_cluster_mask = y == label
            mean_dist = np.mean(dist_mat[i, other_cluster_mask])
            b_i = min(b_i, mean_dist)
        return b_i