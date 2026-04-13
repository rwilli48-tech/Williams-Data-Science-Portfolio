# This module is dedicated to loading a dataset 
import pandas as pd
from pathlib import Path


# Path to the sample_data folder
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

DATASET_DESCRIPTIONS = {
    "monsters.csv": "Dungeons & Dragons monsters dataset (Great for classification models and for testing unscaled data).",
    "titanic.csv": "Titanic survival dataset (Great for seeing how the app handles lots of missing data).",
    "student_performance.csv": "Student performance dataset (Good dataset for both classification and regression models)."
}

def get_dataset_description(filename):
    return DATASET_DESCRIPTIONS.get(filename, "No description available.")

def get_sample_dataset_names():
    """
    Returns a list of available sample dataset filenames in the sample_data folder.
    """
    if not SAMPLE_DATA_DIR.exists():
        return []

    csv_files = sorted([file.name for file in SAMPLE_DATA_DIR.glob("*.csv")])
    return csv_files


def load_uploaded_file(uploaded_file):
    """
    Loads an uploaded CSV file into a pandas DataFrame.
    Returns None if loading fails.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        return None


def load_sample_data(filename):
    """
    Loads a sample CSV file from the sample_data folder.
    Returns None if loading fails.
    """
    try:
        file_path = SAMPLE_DATA_DIR / filename
        df = pd.read_csv(file_path)
        return df
    except Exception:
        return None