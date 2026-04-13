import pandas as pd


def get_basic_info(df):
    """
    Returns basic dataset information as a dictionary.
    """
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": list(df.columns)
    }


def get_data_types(df):
    """
    Returns a DataFrame showing each column and its data type.
    """
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
    return dtypes_df


def get_missing_values(df):
    """
    Returns a DataFrame showing missing value counts and percentages by column.
    """
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().mean() * 100).round(2)

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_count.values,
        "Percent Missing": missing_percent.values
    })

    return missing_df.sort_values(by="Missing Values", ascending=False)


def get_descriptive_stats(df):
    """
    Returns descriptive statistics for all columns.
    """
    return df.describe(include="all").transpose()


def get_duplicate_count(df):
    """
    Returns the number of duplicate rows in the dataset.
    """
    return int(df.duplicated().sum())