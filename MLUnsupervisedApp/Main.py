import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

from utils.evaluation import (
    evaluate_kmeans,
    get_cluster_size_table,
    get_pca_explained_variance_table,
    get_pca_loadings_table
)

from utils.summarizer import (
    get_basic_info,
    get_data_types,
    get_missing_values,
    get_descriptive_stats,
    get_duplicate_count
)

from utils.data_loader import (
    load_uploaded_file,
    load_sample_data,
    get_sample_dataset_names,
    get_dataset_description
)

from utils.preprocessing import (
    get_numeric_columns,
    get_non_numeric_columns,
    create_analysis_df,
    preprocess_features,
    get_feature_summary
)

from utils.modeling import (
    get_available_models,
    get_model,
    get_model_description
)

st.set_page_config(page_title="Unsupervised ML App", layout="wide")

st.title("Unsupervised Machine Learning Playground")
st.write("Upload your own dataset or choose a sample dataset to explore patterns using unsupervised machine learning.")

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
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)

        if df is not None:
            st.success("Dataset uploaded successfully.")
        else:
            st.error("Could not read the uploaded file. Please make sure it is a valid CSV.")

elif data_source == "Use Sample Dataset":
    sample_options = get_sample_dataset_names()

    selected_dataset = st.selectbox("Choose a sample dataset:", sample_options)

    if selected_dataset:
        description = get_dataset_description(selected_dataset)
        st.info(description)

        df = load_sample_data(selected_dataset)

        if df is not None:
            st.success(f"Loaded sample dataset: {selected_dataset}")
        else:
            st.error("Could not load the sample dataset.")

# -----------------------------
# Main app content
# -----------------------------
if df is not None:

    st.subheader("Dataset Preview: First 20 Rows")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Shape")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

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

    st.subheader("Basic Information")
    info_col1, info_col2, info_col3 = st.columns(3)

    info_col1.metric("Rows", basic_info["num_rows"])
    info_col2.metric("Columns", basic_info["num_columns"])
    info_col3.metric("Duplicate Rows", duplicate_count)

    st.subheader("Data Types")
    st.dataframe(dtypes_df, use_container_width=True)

    st.subheader("Missing Values Summary")
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(stats_df, use_container_width=True)

    # -----------------------------
    # Feature Selection
    # -----------------------------
    st.header("3. Select Features for Analysis")

    numeric_columns = get_numeric_columns(df)
    non_numeric_columns = get_non_numeric_columns(df)

    if len(numeric_columns) == 0:
        st.error("This dataset does not contain any numeric columns. Unsupervised models need numeric features.")
    else:
        st.write(
            "Choose the numeric columns you want the unsupervised model to use. "
            "For now, this app uses numeric columns only to keep the analysis clean and reliable."
        )

        selected_features = st.multiselect(
            "Select numeric features:",
            numeric_columns,
            default=numeric_columns
        )

        feature_summary = get_feature_summary(df, selected_features)

        st.subheader("Feature Summary")

        feature_col1, feature_col2, feature_col3 = st.columns(3)
        feature_col1.metric("Numeric Columns Available", feature_summary["numeric_available"])
        feature_col2.metric("Selected Features", feature_summary["selected_features"])
        feature_col3.metric("Non-Numeric Columns Excluded", feature_summary["non_numeric_excluded"])

        with st.expander("Show numeric columns"):
            st.write(feature_summary["numeric_columns"])

        with st.expander("Show non-numeric columns excluded for now"):
            st.write(feature_summary["non_numeric_columns"])

        if len(selected_features) == 0:
            st.warning("Select at least one numeric feature to continue.")
        else:
            analysis_df = create_analysis_df(df, selected_features)

            st.subheader("Selected Feature Preview")
            st.dataframe(analysis_df.head(20), use_container_width=True)

            # -----------------------------
            # Preprocessing
            # -----------------------------
            st.header("4. Preprocess Features")

            st.write(
                "The app fills missing numeric values using the median and then scales the features. "
                "Scaling is important because K-Means and PCA are affected by feature magnitude."
            )

            processed_df, missing_before, missing_after = preprocess_features(analysis_df)

            pre_col1, pre_col2 = st.columns(2)
            pre_col1.metric("Missing Values Before", missing_before)
            pre_col2.metric("Missing Values After", missing_after)

            st.subheader("Preprocessed Feature Preview")
            st.dataframe(processed_df.head(20), use_container_width=True)

            # -----------------------------
            # Model Selection
            # -----------------------------
            st.header("5. Choose Unsupervised Model")

            available_models = get_available_models()

            model_name = st.selectbox(
                "Choose an unsupervised machine learning model:",
                available_models
            )

            model_description = get_model_description(model_name)

            if model_description:
                st.info(model_description)

            # -----------------------------
            # Model Settings
            # -----------------------------
            st.header("6. Adjust Model Settings")

            hyperparams = {}

            if model_name == "K-Means Clustering":
                st.subheader("K-Means Settings")

                hyperparams["n_clusters"] = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=3,
                    step=1
                )

                hyperparams["max_iter"] = st.slider(
                    "Maximum iterations",
                    min_value=100,
                    max_value=1000,
                    value=300,
                    step=50
                )

                hyperparams["random_state"] = st.number_input(
                    "Random state",
                    min_value=0,
                    value=42,
                    step=1
                )

            elif model_name == "Principal Component Analysis":
                st.subheader("PCA Settings")

                max_components = min(len(selected_features), len(processed_df))

                hyperparams["n_components"] = st.slider(
                    "Number of principal components",
                    min_value=2,
                    max_value=max_components,
                    value=min(2, max_components),
                    step=1
                )

            model = get_model(model_name, hyperparams)

            if model is not None:
                st.success(f"{model_name} is ready to run.")
                st.write("Selected hyperparameters:", hyperparams)
            else:
                st.error("Model could not be created.")

            # -----------------------------
            # Run Model
            # -----------------------------
            st.header("7. Run Model")

            run_button = st.button("Run Unsupervised Model")

            if run_button and model is not None:
                try:

                    # -----------------------------
                    # K-Means Results
                    # -----------------------------
                    if model_name == "K-Means Clustering":
                        cluster_labels = model.fit_predict(processed_df)

                        results_df = df.copy()
                        results_df["Cluster"] = cluster_labels

                        st.success("K-Means clustering completed successfully.")

                        st.header("8. Results and Visualizations")

                        kmeans_results = evaluate_kmeans(processed_df, cluster_labels, model)

                        metric_col1, metric_col2 = st.columns(2)

                        if kmeans_results["silhouette_score"] is not None:
                            metric_col1.metric(
                                "Silhouette Score",
                                f"{kmeans_results['silhouette_score']:.3f}"
                            )
                        else:
                            metric_col1.metric("Silhouette Score", "N/A")

                        metric_col2.metric(
                            "Inertia",
                            f"{kmeans_results['inertia']:.3f}"
                        )

                        st.subheader("Cluster Size Table")
                        cluster_size_table = get_cluster_size_table(cluster_labels)
                        st.dataframe(cluster_size_table, use_container_width=True)

                        st.subheader("Dataset with Cluster Labels")
                        st.dataframe(results_df.head(30), use_container_width=True)

                        pca_visual = PCA(n_components=2)
                        pca_array = pca_visual.fit_transform(processed_df)

                        pca_plot_df = pd.DataFrame({
                            "PC1": pca_array[:, 0],
                            "PC2": pca_array[:, 1],
                            "Cluster": cluster_labels
                        })

                        st.subheader("2D Cluster Visualization Using PCA")
                        st.caption(
                            "This plot uses PCA to compress the selected features into two dimensions so the clusters can be visualized."
                        )

                        fig, ax = plt.subplots()
                        scatter = ax.scatter(
                            pca_plot_df["PC1"],
                            pca_plot_df["PC2"],
                            c=pca_plot_df["Cluster"]
                        )
                        ax.set_xlabel("Principal Component 1")
                        ax.set_ylabel("Principal Component 2")
                        ax.set_title("K-Means Clusters Visualized with PCA")
                        fig.colorbar(scatter, ax=ax, label="Cluster")

                        st.pyplot(fig)

                        st.info(
                            "Interpretation: a higher silhouette score usually means the clusters are more clearly separated. "
                            "Lower inertia means observations are closer to their assigned cluster centers, but inertia naturally decreases as the number of clusters increases."
                        )

                    # -----------------------------
                    # PCA Results
                    # -----------------------------
                    elif model_name == "Principal Component Analysis":
                        pca_array = model.fit_transform(processed_df)

                        pca_columns = [f"PC{i+1}" for i in range(model.n_components_)]
                        pca_df = pd.DataFrame(pca_array, columns=pca_columns)

                        st.success("PCA completed successfully.")

                        st.header("8. Results and Visualizations")

                        explained_variance_table = get_pca_explained_variance_table(model)

                        st.subheader("Explained Variance Table")
                        st.dataframe(explained_variance_table, use_container_width=True)

                        st.subheader("Cumulative Explained Variance Plot")

                        fig, ax = plt.subplots()
                        ax.plot(
                            explained_variance_table["Component"],
                            explained_variance_table["Cumulative Explained Variance"],
                            marker="o"
                        )
                        ax.set_xlabel("Principal Component")
                        ax.set_ylabel("Cumulative Explained Variance")
                        ax.set_title("Cumulative Explained Variance by PCA Component")

                        st.pyplot(fig)

                        st.subheader("PCA Results Preview")
                        st.dataframe(pca_df.head(30), use_container_width=True)

                        if model.n_components_ >= 2:
                            st.subheader("2D PCA Scatterplot")

                            fig, ax = plt.subplots()
                            ax.scatter(pca_df["PC1"], pca_df["PC2"])
                            ax.set_xlabel("Principal Component 1")
                            ax.set_ylabel("Principal Component 2")
                            ax.set_title("PCA Scatterplot")

                            st.pyplot(fig)

                        st.subheader("PCA Component Loadings")
                        loadings_table = get_pca_loadings_table(model, selected_features)
                        st.dataframe(loadings_table, use_container_width=True)

                        st.info(
                            "Interpretation: explained variance tells you how much information each principal component preserves. "
                            "Component loadings show which original features contribute most strongly to each component."
                        )

                except Exception as e:
                    st.error(f"An error occurred while running the model: {e}")

else:
    st.info("Please upload a CSV file or select a sample dataset to continue.")