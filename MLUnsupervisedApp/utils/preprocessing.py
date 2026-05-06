import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def get_numeric_columns(df):
    """
    Returns a list of numeric columns in the dataset.
    """
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_non_numeric_columns(df):
    """
    Returns a list of non-numeric columns in the dataset.
    """
    return df.select_dtypes(exclude=["number"]).columns.tolist()


def create_analysis_df(df, selected_features):
    """
    Creates a DataFrame using only the selected features.
    """
    return df[selected_features].copy()


def preprocess_features(analysis_df):
    """
    Imputes missing numeric values using the median and scales the data.

    Returns:
        processed_df: scaled DataFrame ready for unsupervised learning
        missing_before: number of missing values before preprocessing
        missing_after: number of missing values after preprocessing
    """
    missing_before = int(analysis_df.isnull().sum().sum())

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    imputed_array = imputer.fit_transform(analysis_df)
    scaled_array = scaler.fit_transform(imputed_array)

    processed_df = pd.DataFrame(
        scaled_array,
        columns=analysis_df.columns
    )

    missing_after = int(processed_df.isnull().sum().sum())

    return processed_df, missing_before, missing_after


def get_feature_summary(df, selected_features):
    """
    Returns a summary of selected and excluded features.
    """
    numeric_columns = get_numeric_columns(df)
    non_numeric_columns = get_non_numeric_columns(df)

    return {
        "numeric_available": len(numeric_columns),
        "selected_features": len(selected_features),
        "non_numeric_excluded": len(non_numeric_columns),
        "numeric_columns": numeric_columns,
        "non_numeric_columns": non_numeric_columns
    }