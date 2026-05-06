import pandas as pd


def validate_dataset_for_task(df, target_column, task_type):
    """
    Validates whether the dataset is suitable for the selected ML task.

    Returns:
        {
            "errors": [...],
            "warnings": [...]
        }
    """
    errors = []
    warnings = []

    # Basic dataset checks
    if df is None or df.empty:
        errors.append("The dataset is empty.")
        return {"errors": errors, "warnings": warnings}

    if target_column not in df.columns:
        errors.append("The selected target column was not found in the dataset.")
        return {"errors": errors, "warnings": warnings}

    # Drop missing target values for validation purposes
    cleaned_df = df.dropna(subset=[target_column])

    if cleaned_df.empty:
        errors.append("All rows have missing target values after removing missing targets.")
        return {"errors": errors, "warnings": warnings}

    y = cleaned_df[target_column]
    X = cleaned_df.drop(columns=[target_column])

    # Target checks
    if y.nunique() <= 1:
        errors.append("The target column must contain at least two unique values.")

    if X.shape[1] == 0:
        errors.append("No feature columns remain after removing the target column.")

    # Task-specific checks
    if task_type == "Regression":
        if not pd.api.types.is_numeric_dtype(y):
            errors.append("Regression requires a numeric target column.")

    elif task_type == "Classification":
        num_classes = y.nunique()

        if num_classes > 20:
            warnings.append(
                f"The selected classification target has {num_classes} unique classes. "
                "This may not be appropriate for basic classification models."
            )

        class_distribution = y.value_counts(normalize=True)
        if not class_distribution.empty and class_distribution.iloc[0] > 0.9:
            warnings.append(
                "The target is highly imbalanced. One class makes up more than 90% of the data."
            )

    # General warnings
    if len(cleaned_df) < 30:
        warnings.append(
            "The dataset has fewer than 30 usable rows. Model performance may be unstable."
        )

    if X.shape[1] > len(cleaned_df):
        warnings.append(
            "The dataset has more features than rows. This increases the risk of overfitting."
        )

    total_missing = cleaned_df.isnull().sum().sum()
    if total_missing > 0:
        warnings.append(
            "The dataset contains missing values. The app will apply basic preprocessing, "
            "but results may still be affected."
        )

    if X.select_dtypes(exclude=["number"]).shape[1] > 0:
        warnings.append(
            "The dataset contains categorical columns. These will be one-hot encoded automatically."
        )

    return {"errors": errors, "warnings": warnings}