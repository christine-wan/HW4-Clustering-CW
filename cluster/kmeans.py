import numpy as np
from scipy.spatial.distance import cdist
import warnings


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100, verbose: bool = False):
        """
        Initializes the KMeans clustering model.

        Inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("Invalid k: Must be an integer >= 1")
        if tol < 0:
            raise ValueError("tol must be a positive value!")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")

        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.centers = None
        self.error = None

    def _initialize_centroids(self, mat: np.ndarray) -> np.ndarray:
        """
        Initializes centroids using the KMeans++ algorithm.
        """
        rng = np.random.default_rng()

        if self.k >= mat.shape[0]:  # If k is close to n, return unique points
            unique_points = np.unique(mat, axis=0)
            return unique_points[:self.k]

        first_centroid = mat[rng.choice(mat.shape[0])]
        centroids = [first_centroid]

        for _ in range(1, self.k):
            dists = np.min(cdist(mat, np.array(centroids)), axis=1)
            probs = dists / dists.sum()
            new_centroid = mat[rng.choice(mat.shape[0], p=probs)]
            centroids.append(new_centroid)

        return np.array(centroids)

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        """
        if mat.shape[0] < self.k:
            raise ValueError("Data must include at least k data points")
        if mat.ndim != 2:
            raise ValueError("Input data must be a 2D matrix")

        self.centers = self._initialize_centroids(mat)
        iters = 0
        prev_error = float("inf")

        while iters < self.max_iter:
            dists = cdist(mat, self.centers, 'euclidean')
            min_dists = np.argmin(dists, axis=1)  # Assign points to clusters

            # Compute new centroids
            one_hot_matrix = np.eye(self.k)[min_dists]
            sum_values = np.dot(one_hot_matrix.T, mat)
            group_counts = np.sum(one_hot_matrix, axis=0)

            # Handle empty clusters
            empty_clusters = np.where(group_counts == 0)[0]
            if empty_clusters.size > 0:
                unassigned_points = np.setdiff1d(np.arange(mat.shape[0]), min_dists)
                rng = np.random.default_rng()

                for i in empty_clusters:
                    if unassigned_points.size > 0:
                        new_idx = rng.choice(unassigned_points)
                        sum_values[i] = mat[new_idx]  # Assign a random unassigned point
                        group_counts[i] = 1
                        unassigned_points = np.setdiff1d(unassigned_points, [new_idx])
                    else:
                        sum_values[i] = mat[rng.choice(mat.shape[0])]  # Fallback to random reassignment

                    if self.verbose:
                        print(f"Reassigned empty cluster {i} to new point.")

            non_empty = group_counts > 0
            self.centers = np.where(non_empty[:, None], sum_values / group_counts[:, None], self.centers)

            # Compute error (sum of squared distances)
            error = np.sum(np.min(dists, axis=1) ** 2)
            self.error = error

            # Check for convergence
            if abs(prev_error - error) / (prev_error + 1e-10) < self.tol:  # Avoid division by zero
                if self.verbose:
                    print(f"Converged after {iters} iterations with error {error:.6f}")
                break

            prev_error = error
            iters += 1

        if iters == self.max_iter and self.verbose:
            warnings.warn(f"Failed to converge after {self.max_iter} iterations")

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points.
        """
        if self.centers is None:
            raise ValueError("The model needs to be fitted before making predictions.")
        if mat.shape[1] != self.centers.shape[1]:
            raise ValueError("Input data must have the same number of features as the data used to fit the model.")

        dists = cdist(mat, self.centers, 'euclidean')
        return np.argmin(dists, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model.
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.
        """
        return self.centers
