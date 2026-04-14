# Machine Learning Streamlit App

## Project Overview

This project is an miniature exploration of what I've learned so far in regards to Machine Learning models. 
At its core, it is a playground for users to upload their own csv datasets or use one of the three trial datasets, 
configure a supervised learning model, and evaluate that model through numeric performance metrics and explanatory visuals. At its periphery,
it is an interactive summary of the models that I have studied in my Intro to Data Science Class. 

The app is designed to guide users through the full machine learning workflow, including:
- Data exploration and summary
- Target variable selection
- Model selection and hyperparameter tuning
- Model training and evaluation

Users can experiment with different models and settings to better understand how machine learning models behave on different datasets.

---

## Features

### Data Input
- Upload custom CSV datasets
- Select from built-in sample datasets

### Dataset Exploration
- Dataset preview
- Summary statistics
- Missing value analysis
- Data type inspection

### Prediction Setup
- Select target variable
- Choose prediction type (Classification or Regression)
- Configure train/test split

### Machine Learning Models
The app supports the following supervised learning models:

- **Linear Regression** (Regression)
- **Logistic Regression** (Classification)
- **Decision Tree** (Classification & Regression)
- **K-Nearest Neighbors (KNN)** (Classification & Regression)

### Hyperparameter Tuning
Users can adjust key hyperparameters using interactive controls:
- Logistic Regression: regularization strength (`C`), max iterations
- Decision Tree: max depth, minimum samples per split
- KNN: number of neighbors (`k`)

### Model Evaluation

#### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC (for binary classification)
- Confusion Matrix
- ROC Curve visualization

#### Regression Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Actual vs Predicted plot

### Dataset Health Check
- Detects potential issues such as:
  - Missing target values
  - Imbalanced classes
  - Small dataset size
  - Too many features relative to observations
- Provides warnings and prevents training when critical issues are detected

### Feature Importance
- Displays feature importance for Decision Tree models

---

## Installation & Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
conda create -n ml_app python=3.10
conda activate ml_app
pip install -r requirements.txt
streamlit run Main.py
