# Machine Learning Explorer App

## Project Overview

## Project Overview

This project is a small-scale exploration of what I’ve learned so far about unsupervised machine learning and data-driven pattern discovery. At its core, it is a tool for users to upload their own CSV datasets and uncover hidden structure within the data through clustering and segmentation techniques. More broadly, it acts as an interactive reflection of the unsupervised learning concepts covered in my Intro to Data Science course.

Rather than predicting a target variable, the app is designed to help users identify natural groupings and relationships in their data. It walks through key steps of the unsupervised learning workflow, including:
- Data exploration and summary
- Feature selection and preprocessing
- Clustering model selection (e.g., K-Means)
- Evaluation of cluster quality and structure
- Visualization of grouped data

Users can experiment with different configurations, such as the number of clusters, to see how the underlying structure of their dataset changes. The goal is to build intuition around how unsupervised models detect patterns without predefined labels.
---

## Features

### 1. Data Upload & Preview
- Upload CSV datasets directly into the app
- Automatically display dataset structure and summary statistics
- Identify feature types (numeric vs categorical)

### 2. Data Cleaning
- Handles missing values in the target variable
- Separates features (X) and target (y)
- Prepares data for modeling without requiring manual preprocessing

### 3. Exploratory Data Analysis
- Summary statistics for all variables
- Target variable insights:
  - Data type
  - Number of unique values
  - Missing values

### 4. Machine Learning Models
The app supports multiple models depending on the problem type:

#### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

#### Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### 5. Model Evaluation
- Train/test split applied automatically
- Performance metrics displayed based on model type:

**Classification Metrics**
- Accuracy
- Precision
- ROC-AUC

**Regression Metrics**
- Mean Squared Error (MSE)
- R² Score

### 6. Visualization
- Actual vs Predicted plots (regression)
- Model performance summaries
- Clean, interpretable outputs for quick insights

### 7. Streamlit Interface
- Sidebar navigation for model and feature selection
- Interactive and intuitive layout
- Designed for quick experimentation

---

## How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name