import numpy as np
import pytest
from sklearn.metrics import silhouette_score
from cluster import silhouette, kmeans, utils


def test_silhouette_true():
    # Using utils to generate clusters
    true_clusters, true_labels = utils.make_clusters(n=1000, k=3)

    # Initialize silhouette scorer & get scores
    s = silhouette.Silhouette()
    scores = s.score(true_clusters, true_labels)

    # Compare the mean of the scores with sklearn's silhouette_score function
    assert np.isclose(np.mean(scores), silhouette_score(true_clusters, true_labels))


def test_silhouette_pred():
    # Using utils to generate clusters
    true_clusters, true_labels = utils.make_clusters(n=1000, k=3)

    # Fit KMeans model
    kmeans_model = kmeans.KMeans(k=3)
    kmeans_model.fit(true_clusters)

    # Predict labels
    predicted_labels = kmeans_model.predict(true_clusters)

    # Initialize silhouette scorer & get scores
    s = silhouette.Silhouette()
    scores = s.score(true_clusters, predicted_labels)

    # Compare the mean of the scores with sklearn's silhouette_score function
    assert np.isclose(np.mean(scores), silhouette_score(true_clusters, predicted_labels))


def test_silhouette_low_score():
    # Using utils to generate loosely clustered data
    loose_clusters, loose_labels = utils.make_clusters(n=1000, k=3, scale=2)

    # Fit KMeans model
    kmeans_model = kmeans.KMeans(k=3)
    kmeans_model.fit(loose_clusters)

    # Predict the labels
    predicted_labels = kmeans_model.predict(loose_clusters)

    # Initialize silhouette scorer & get scores
    s = silhouette.Silhouette()
    scores = s.score(loose_clusters, predicted_labels)

    # Ensure the average silhouette score is low for poorly clustered data
    assert np.mean(scores) < 0.5


def test_silhouette_high_score():
    # Using utils to generate tightly clustered data
    tight_clusters, tight_labels = utils.make_clusters(n=1000, k=3, scale=0.1)

    # Fit KMeans model
    kmeans_model = kmeans.KMeans(k=3)
    kmeans_model.fit(tight_clusters)

    # Predict the labels
    predicted_labels = kmeans_model.predict(tight_clusters)

    # Initialize silhouette scorer & get scores
    s = silhouette.Silhouette()
    scores = s.score(tight_clusters, predicted_labels)

    # Ensure the average silhouette score is high for tightly clustered data
    assert np.mean(scores) >= 0.8


def test_silhouette_invalid_data():
    # Test for empty data
    s = silhouette.Silhouette()
    with pytest.raises(ValueError):
        s.score(np.array([]), np.array([0, 1, 1]))  # Empty X

    with pytest.raises(ValueError):
        s.score(np.array([[1, 2], [3, 4]]), np.array([]))  # Empty y

    # Test for mismatched dimensions
    with pytest.raises(ValueError):
        s.score(np.array([[1, 2], [3, 4]]), np.array([0]))


def test_silhouette_nan_handling():
    """Test if Silhouette correctly handles NaN values for single-point clusters."""
    # Generate a small dataset with 3 points and 2 clusters
    X, y = utils.make_clusters(n=3, k=2)

    # Ensure one cluster ends up with only a single point
    if np.sum(y == 0) == 1:
        y[1] = 1
    else:
        y[0] = 2

    s = silhouette.Silhouette()
    scores = s.score(X, y)

    assert np.isnan(scores[0])  # Expect NaN for the point in the single-member cluster
