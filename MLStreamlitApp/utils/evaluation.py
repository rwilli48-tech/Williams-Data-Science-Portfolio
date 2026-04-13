import numpy as np
import pandas as pd

# For the ROC curve
import matplotlib.pyplot as plt

#Needed libraries
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve
)


def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Returns a dictionary of classification metrics.
    ROC AUC is included when probability scores are available for binary classification.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": None
    }

    # Only compute ROC AUC for binary classification when y_prob is provided
    unique_classes = np.unique(y_true)
    if y_prob is not None and len(unique_classes) == 2:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            results["roc_auc"] = None

    return results


def get_confusion_matrix_df(y_true, y_pred):
    """
    Returns confusion matrix as a labeled pandas DataFrame.
    """
    labels = sorted(pd.Series(y_true).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )
    return cm_df


def evaluate_regression(y_true, y_pred):
    """
    Returns a dictionary of regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred)
    }

def plot_roc_curve(y_true, y_prob):
    """
    Creates a ROC curve plot for binary classification.
    Returns a matplotlib figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    return fig


def get_feature_importance_df(pipeline, feature_names):
    """
    Extracts feature importances from a trained decision tree model.
    Returns a DataFrame sorted by importance.
    """
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    return importance_df