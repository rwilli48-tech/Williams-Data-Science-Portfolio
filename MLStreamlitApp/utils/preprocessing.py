import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(df, target_column):
    """
    Splits a DataFrame into features (X) and target (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_column_types(X):
    """
    Identifies numeric and categorical feature columns.
    """
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    return numeric_columns, categorical_columns


def get_target_summary(y):
    """
    Returns a small summary of the target column.
    """
    return {
        "name": y.name,
        "dtype": str(y.dtype),
        "num_missing": int(y.isnull().sum()),
        "num_unique": int(y.nunique())
    }


def drop_missing_target_rows(df, target_column):
    """
    Drops rows where the target is missing.
    Returns the cleaned DataFrame and number of dropped rows.
    """
    original_rows = len(df)
    cleaned_df = df.dropna(subset=[target_column]).copy()
    dropped_rows = original_rows - len(cleaned_df)

    return cleaned_df, dropped_rows


def basic_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Performs a simple train/test split.
    Uses stratify=y for classification if appropriate.
    """
    stratify = None

    # Safe default for classification-like targets with a manageable number of classes
    if y.nunique() > 1 and y.nunique() <= 20:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test