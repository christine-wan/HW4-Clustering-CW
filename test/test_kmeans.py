import pytest
import numpy as np
from cluster import kmeans, utils


# --- KMeans Initialization Tests ---
def test_kmeans_initialization():
    # Test valid initialization
    kmeans_test = kmeans.KMeans(k=3, tol=1e-4, max_iter=200)
    assert kmeans_test.k == 3
    assert kmeans_test.tol == 1e-4
    assert kmeans_test.max_iter == 200

    # Test invalid k value (less than 1)
    with pytest.raises(ValueError):
        kmeans.KMeans(k=0)

    # Test invalid tol (negative)
    with pytest.raises(ValueError):
        kmeans.KMeans(k=3, tol=-1)

    # Test invalid max_iter (less than 1)
    with pytest.raises(ValueError):
        kmeans.KMeans(k=3, max_iter=0)


# --- KMeans Fit and Predict Tests ---
def test_kmeans_fit():
    # Using utils to generate clusters
    X, _ = utils.make_clusters(n=6, k=2)
    kmeans_model = kmeans.KMeans(k=2)
    kmeans_model.fit(X)

    assert kmeans_model.centers is not None
    assert kmeans_model.centers.shape == (2, 2)  # 2 centroids, each with 2 features


def test_kmeans_predict():
    # Using utils to generate clusters
    X, _ = utils.make_clusters(n=6, k=2)
    kmeans_model = kmeans.KMeans(k=2)
    kmeans_model.fit(X)

    labels = kmeans_model.predict(X)
    assert len(labels) == len(X)
    assert np.all(np.isin(np.unique(labels), [0, 1]))  # Clusters are labeled 0 or 1

# --- Additional Test: get_centroids ---
def test_kmeans_get_centroids():
    X, _ = utils.make_clusters(n=10, k=3)
    kmeans_model = kmeans.KMeans(k=3)
    kmeans_model.fit(X)
    centroids = kmeans_model.get_centroids()
    assert centroids.shape == (3, X.shape[1])  # k centroids, same feature dimension as X

# --- KMeans Edge Case Tests ---
def test_kmeans_edge_cases():
    # Generate test data using utils
    X, _ = utils.make_clusters(n=6, k=2)
    kmeans_model = kmeans.KMeans(k=2)
    kmeans_model.fit(X)

    # Test when k equals the number of data points
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    kmeans_model = kmeans.KMeans(k=5)
    kmeans_model.fit(X)
    labels = kmeans_model.predict(X)
    assert len(np.unique(labels)) == 5  # Each point should be its own cluster

    # Test for empty dataset
    X_empty = np.empty((0, 2))  # No points
    with pytest.raises(ValueError):
        kmeans_model.fit(X_empty)

    # Test for very high-dimensional data (many features)
    X_highdim = np.random.rand(10, 1000)  # 10 points, 1000 features
    kmeans_model = kmeans.KMeans(k=3)
    kmeans_model.fit(X_highdim)
    labels = kmeans_model.predict(X_highdim)
    assert len(labels) == 10