import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate_kmeans(processed_df, cluster_labels, model):
    """
    Returns K-Means evaluation metrics.
    """
    n_clusters = len(set(cluster_labels))

    if n_clusters > 1 and n_clusters < len(processed_df):
        silhouette = silhouette_score(processed_df, cluster_labels)
    else:
        silhouette = None

    return {
        "silhouette_score": silhouette,
        "inertia": model.inertia_
    }


def get_cluster_size_table(cluster_labels):
    """
    Returns a table showing the number of observations in each cluster.
    """
    cluster_table = pd.Series(cluster_labels).value_counts().sort_index()

    return pd.DataFrame({
        "Cluster": cluster_table.index,
        "Count": cluster_table.values
    })


def get_pca_explained_variance_table(model):
    """
    Returns PCA explained variance by component.
    """
    return pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(model.explained_variance_ratio_))],
        "Explained Variance Ratio": model.explained_variance_ratio_,
        "Cumulative Explained Variance": model.explained_variance_ratio_.cumsum()
    })


def get_pca_loadings_table(model, feature_names):
    """
    Returns PCA component loadings.
    """
    return pd.DataFrame(
        model.components_,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(model.n_components_)]
    )