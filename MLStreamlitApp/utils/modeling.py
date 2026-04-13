from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

MODEL_DESCRIPTIONS = {
    "Linear Regression": "Assumes a linear relationship to predict the next value. Strictly a regression model.",
    "Logistic Regression": "Classification model that predicts class probabilities. Strictly a classification model.",
    "Decision Tree": "Makes predictions by splitting the data into branches according to feature values. Used for both classification and regression problems.",
    "K-Nearest Neighbors": "Makes predictions based on proximity of data points in the training set. Used for both classification and regression problems."
}

def get_model_description(model_name):
    """
    Returns a short explanation of the selected model.
    """
    return MODEL_DESCRIPTIONS.get(model_name)

def get_available_models(task_type):
    """
    Returns a list of model names available for the selected task type.
    """
    if task_type == "Classification":
        return [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors"
        ]
    elif task_type == "Regression":
        return [
            "Linear Regression",
            "Decision Tree",
            "K-Nearest Neighbors"
        ]
    return []


def get_model(model_name, task_type, hyperparams):
    """
    Returns an instantiated sklearn model based on model name, task type,
    and hyperparameters.
    """
    if task_type == "Classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(
                C=hyperparams.get("C", 1.0),
                max_iter=hyperparams.get("max_iter", 200)
            )

        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=hyperparams.get("max_depth", None),
                min_samples_split=hyperparams.get("min_samples_split", 2),
                random_state=42
            )

        elif model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(
                n_neighbors=hyperparams.get("n_neighbors", 5)
            )

    elif task_type == "Regression":
        if model_name == "Linear Regression":
            return LinearRegression()

        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(
                max_depth=hyperparams.get("max_depth", None),
                min_samples_split=hyperparams.get("min_samples_split", 2),
                random_state=42
            )

        elif model_name == "K-Nearest Neighbors":
            return KNeighborsRegressor(
                n_neighbors=hyperparams.get("n_neighbors", 5)
            )

    return None