import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic-1.csv') 

st.write("**Summary Statistics**")
st.dataframe(df.describe())


st.write("heatmap of Missing Values")

fig, ax = plt.subplots()

sns.heatmap(df.isnull(), cmap = "viridis", cbar = False)

st.pyplot(fig)


column = st.selectbox("Choose a column to fill",
            df.select_dtypes(include = ['number']).columns)

method = st.radio("Choose a Method",
         ["Original DF", "Drop Rows", "Drop Columns (>50% Missing)",
          "Impute Mean", "Impute Median", "Impute Zero"])


df_clean = df.copy()

if method == "Original DF":
    pass
elif method == "Drop Rows":
    df_clean = df_clean.dropna()
elif method == "Drop Columns (>50% Missing)":
    # Drop columns with 50% or more missing 
    df_clean = df_clean.drop(column = df_clean.columns[df_clean.isnull().mean() > 0.5])
elif method == "Impute Mean":
    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
elif method == "Impute Median":
    df_clean[column] = df_clean[column].fillna(df_clean[column].median())
elif method == "Impute Zero": 
    df_clean[column] = df_clean[column].fillna(0)



col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Data Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True)
    plt.title(f"Original Distribution of {column}")
    st.pyplot(fig)
    st.subheader(f"{column}'s Original Stats")
    st.write(df[column].describe())


with col2:
    st.subheader("Cleaned Data Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_clean[column], kde=True)
    plt.title(f"Distribution of {column} after {method}")
    st.pyplot(fig)