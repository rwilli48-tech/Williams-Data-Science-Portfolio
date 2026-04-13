import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.evaluation import evaluate_classification, get_confusion_matrix_df, evaluate_regression, plot_roc_curve, get_feature_importance_df
from utils.summarizer import get_basic_info, get_data_types, get_missing_values, get_descriptive_stats, get_duplicate_count
from utils.data_loader import load_uploaded_file, load_sample_data, get_sample_dataset_names, get_dataset_description
from utils.preprocessing import split_features_target, get_column_types, get_target_summary, drop_missing_target_rows, basic_train_test_split
from utils.modeling import get_available_models, get_model, get_model_description
from utils.validation import validate_dataset_for_task

st.set_page_config(page_title="ML Streamlit App", layout="wide")

st.title("Machine Learning Playground")
st.write("Upload your own dataset or choose a sample dataset to begin.")

# -----------------------------
# Data source selection
# -----------------------------
st.header("1. Load Data")

data_source = st.radio(
    "Choose your data source:",
    ["Upload CSV", "Use Sample Dataset"],
    horizontal=True
)

df = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)

        if df is not None:
            st.success("Dataset uploaded successfully.")
        else:
            st.error("Could not read the uploaded file. Please make sure it is a valid CSV.")

elif data_source == "Use Sample Dataset":
    sample_options = get_sample_dataset_names()

    selected_dataset = st.selectbox(
        "Choose a sample dataset:",
        sample_options
    )

    if selected_dataset:
        #Sample Data Description Before Loading
        description = get_dataset_description(selected_dataset)
        st.info(description)

        df = load_sample_data(selected_dataset)

        if df is not None:
            st.success(f"Loaded sample dataset: {selected_dataset}")
        else:
            st.error("Could not load the sample dataset.")

# -----------------------------
# Display dataset preview 
# -----------------------------
if df is not None:
    st.subheader("Dataset Preview: First 20 rows")
    st.dataframe(df.head(20))

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Attributes")
    st.write(list(df.columns))

    # -----------------------------
    # Dataset Summary
    # -----------------------------
    st.header("2. Dataset Summary")

    basic_info = get_basic_info(df)
    duplicate_count = get_duplicate_count(df)
    dtypes_df = get_data_types(df)
    missing_df = get_missing_values(df)
    stats_df = get_descriptive_stats(df)

    # Basic info
    st.subheader("Basic Information")
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", basic_info["num_rows"])
    col2.metric("Columns", basic_info["num_columns"])
    col3.metric("Duplicate Rows", duplicate_count)


    # Data types
    st.subheader("Data Types")
    st.dataframe(dtypes_df, use_container_width=True)

    # Missing values
    st.subheader("Missing Values Summary")
    st.dataframe(missing_df, use_container_width=True)

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(stats_df, use_container_width=True)

        # -----------------------------
    # Model Setup
    # -----------------------------
    st.header("3. Configure Prediction Task")

    target_column = st.selectbox(
        "Select the target column:",
        df.columns
    )

    task_type = st.radio(
        "Select prediction type:",
        ["Classification", "Regression"],
        horizontal=True
    )

    test_size = st.slider(
        "Select test set size:",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    # Clean rows with missing target values
    cleaned_df, dropped_rows = drop_missing_target_rows(df, target_column)

    if dropped_rows > 0:
        st.warning(f"Dropped {dropped_rows} rows because the target column had missing values.")

    # Split into X and y
    X, y = split_features_target(cleaned_df, target_column)

    # Identify feature types
    numeric_columns, categorical_columns = get_column_types(X)

    # Target summary
    target_summary = get_target_summary(y)

    st.subheader("Target Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Name", target_summary["name"])
    col2.metric("Target Type", target_summary["dtype"])
    col3.metric("Missing in Target", target_summary["num_missing"])
    col4.metric("Unique Values", target_summary["num_unique"])

    st.subheader("Feature Summary")
    col5, col6, col7 = st.columns(3)
    col5.metric("Total Features", X.shape[1])
    col6.metric("Numeric Features", len(numeric_columns))
    col7.metric("Categorical Features", len(categorical_columns))

    with st.expander("Show Numeric Feature Names"):
        st.write(numeric_columns)

    with st.expander("Show Categorical Feature Names"):
        st.write(categorical_columns)

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = basic_train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        st.subheader("Train/Test Split")
        split_col1, split_col2 = st.columns(2)
        split_col1.metric("Training Rows", len(X_train))
        split_col2.metric("Testing Rows", len(X_test))

    except Exception as e:
        st.error(f"Could not split the dataset: {e}")
       
    # -----------------------------
    # Dataset Validation
    # -----------------------------
    st.header("4. Dataset Health Check")

    validation_results = validate_dataset_for_task(cleaned_df, target_column, task_type)
    validation_errors = validation_results["errors"]
    validation_warnings = validation_results["warnings"]

    if not validation_errors and not validation_warnings:
        st.success("This dataset looks suitable for the selected machine learning task.")

    for error in validation_errors:
        st.error(error)

    for warning in validation_warnings:
        st.warning(warning)

    can_train = len(validation_errors) == 0
    
    # -----------------------------
    # Model Selection
    # -----------------------------
    st.header("5. Select Model")

    available_models = get_available_models(task_type)

    model_name = st.selectbox(
        "Choose a machine learning model:",
        available_models
    )

    model_description = get_model_description(model_name)

    if model_description:
        st.info(model_description)
    hyperparams = {}

    # Logistic Regression hyperparameters
    if model_name == "Logistic Regression":
        st.subheader("Logistic Regression Settings")

        hyperparams["C"] = st.slider(
            "Regularization strength inverse (C)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01
        )

        hyperparams["max_iter"] = st.slider(
            "Maximum iterations",
            min_value=100,
            max_value=1000,
            value=200,
            step=50
        )

    # Decision Tree hyperparameters
    elif model_name == "Decision Tree":
        st.subheader("Decision Tree Settings")

        max_depth_option = st.checkbox("Limit max depth", value=False)

        if max_depth_option:
            hyperparams["max_depth"] = st.slider(
                "Max depth",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )
        else:
            hyperparams["max_depth"] = None

        hyperparams["min_samples_split"] = st.slider(
            "Minimum samples required to split",
            min_value=2,
            max_value=20,
            value=2,
            step=1
        )

    # KNN hyperparameters
    elif model_name == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors Settings")

        hyperparams["n_neighbors"] = st.slider(
            "Number of neighbors (k)",
            min_value=1,
            max_value=25,
            value=5,
            step=1
        )

    # Linear Regression has no main beginner hyperparameters
    elif model_name == "Linear Regression":
        st.subheader("Linear Regression Settings")
        st.info("Linear Regression is using default settings.")

    # Build model
    model = get_model(model_name, task_type, hyperparams)

    if model is not None:
        st.success(f"{model_name} is ready to train.")
        st.write("Selected hyperparameters:", hyperparams if hyperparams else "Default settings")
    else:
        st.error("Model could not be created.")
    
        # -----------------------------
    # Train and Evaluate
    # -----------------------------
    st.header("6. Train and Evaluate Model")

    if not can_train:
        st.info("Resolve the dataset errors above before training a model.")

    train_button = st.button("Train Model", disabled=not can_train)

    if train_button:
        try:
            # Decide whether scaling is needed
            scale_numeric = model_name in ["Logistic Regression", "K-Nearest Neighbors"]

            # Numeric preprocessing
            if scale_numeric:
                numeric_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
            else:
                numeric_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ])

            # Categorical preprocessing
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Combine preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_columns),
                    ("cat", categorical_transformer, categorical_columns)
                ]
            )

            # Full pipeline
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # Fit model
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)

            st.success("Model trained successfully.")

            # ---------------------------------
            # Classification results
            # ---------------------------------
            if task_type == "Classification":
                y_prob = None

                # Get probabilities if available
                if hasattr(pipeline, "predict_proba"):
                    try:
                        prob_array = pipeline.predict_proba(X_test)
                        if prob_array.shape[1] == 2:
                            y_prob = prob_array[:, 1]
                    except Exception:
                        y_prob = None

                results = evaluate_classification(y_test, y_pred, y_prob)

                st.subheader("Classification Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']:.3f}")
                col2.metric("Precision", f"{results['precision']:.3f}")
                col3.metric("Recall", f"{results['recall']:.3f}")
                col4.metric("F1 Score", f"{results['f1']:.3f}")

                if results["roc_auc"] is not None:
                    st.metric("ROC AUC", f"{results['roc_auc']:.3f}")

                    st.subheader("ROC Curve")
                    roc_fig = plot_roc_curve(y_test, y_prob)
                    st.pyplot(roc_fig)

                st.subheader("Confusion Matrix")
                cm_df = get_confusion_matrix_df(y_test, y_pred)
                st.dataframe(cm_df, use_container_width=True)

            # ---------------------------------
            # Regression results
            # ---------------------------------
            elif task_type == "Regression":
                results = evaluate_regression(y_test, y_pred)

                st.subheader("Regression Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{results['mae']:.3f}")
                col2.metric("RMSE", f"{results['rmse']:.3f}")
                col3.metric("R²", f"{results['r2']:.3f}")

                st.subheader("Actual vs Predicted")
                st.caption("Points closer to a 45 degree diagonal line indicate better predictions.")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted")

                st.pyplot(fig)
            # ---------------------------------
            # Feature Importance for Decision Trees
            # ---------------------------------
            if model_name == "Decision Tree":
                try:
                    preprocessor_fitted = pipeline.named_steps["preprocessor"]

                    transformed_feature_names = preprocessor_fitted.get_feature_names_out()

                    importance_df = get_feature_importance_df(
                        pipeline,
                        transformed_feature_names
                    )

                    if importance_df is not None:
                        st.subheader("Feature Importance")

                        top_n = min(15, len(importance_df))
                        top_features = importance_df.head(top_n)

                        st.dataframe(top_features, use_container_width=True)

                        fig, ax = plt.subplots()
                        ax.barh(
                            top_features["Feature"][::-1],
                            top_features["Importance"][::-1]
                        )
                        ax.set_xlabel("Importance")
                        ax.set_ylabel("Feature")
                        ax.set_title("Top Feature Importances")

                        st.pyplot(fig)

                except Exception as e:
                    st.warning(f"Could not display feature importance: {e}")
        except Exception as e:
            st.error(f"An error occurred during training or evaluation: {e}")
else:
    st.info("Please upload a CSV file or select a sample dataset to continue.")