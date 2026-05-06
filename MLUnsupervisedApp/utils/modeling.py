from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


MODEL_DESCRIPTIONS = {
    "K-Means Clustering": (
        "Groups observations into clusters based on similarity. "
        "Useful for discovering natural groupings in the data."
    ),
    "Principal Component Analysis": (
        "Reduces many numeric features into fewer components while preserving as much variation as possible. "
        "Useful for visualization and dimensionality reduction."
    )
}


def get_available_models():
    """
    Returns the available unsupervised learning models.
    """
    return [
        "K-Means Clustering",
        "Principal Component Analysis"
    ]


def get_model_description(model_name):
    """
    Returns a short explanation of the selected model.
    """
    return MODEL_DESCRIPTIONS.get(model_name)


def get_model(model_name, hyperparams):
    """
    Returns an instantiated unsupervised sklearn model.
    """
    if model_name == "K-Means Clustering":
        return KMeans(
            n_clusters=hyperparams.get("n_clusters", 3),
            max_iter=hyperparams.get("max_iter", 300),
            random_state=hyperparams.get("random_state", 42),
            n_init=10
        )

    elif model_name == "Principal Component Analysis":
        return PCA(
            n_components=hyperparams.get("n_components", 2)
        )

    return None