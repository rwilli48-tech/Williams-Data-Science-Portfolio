import streamlit as st
# similar to having a single hashtag in Markdown (the highest title)
st.title("Hello, streamlit!")
st.markdown("# Hello, streamlit!")
#these two are the same

st.write("This is my first Streamlit app.")

if st.button("Click me!"):
    st.write("Hey you clicked the button!")
else:
    st.write("Click the button and see what happens")

### Loading CSV file
import pandas as pd

st.subheader("Exploring our dataset")

# load in the csv file

df = pd.read_csv("data/sample_data.csv")

st.write("Here's our data!")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique(), index = None)

filtered_df = df[df["City"] == city]

st.write(f"People in {city}")
st.dataframe(filtered_df)

## bar chart 
st.bar_chart(df["Salary"])

import seaborn as sns 

#box plot of earnings distribution among cities 

box_plot1 = sns.boxplot(x = df["City"], y = df["Salary"])

st.pyplot(box_plot1.get_figure())