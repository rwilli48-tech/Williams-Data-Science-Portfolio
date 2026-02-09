# App Idea: general anayltical data searchable by Game/Genre/Publisher/Platform
import zipfile
import pandas as pd
import streamlit as st
import altair as alt

# Data from Kaggle https://www.kaggle.com/datasets/gregorut/videogamesales
ZIP_PATH = "game_sales_data/vgsales.csv.zip"
CSV_NAME = "vgsales.csv"

#Streamlit title (use set_page_config to fit charts and title)
st.set_page_config(page_title="Video Game Sales Explorer", layout="wide")

# Data loading (use cache to avoid problems with zip) 
@st.cache_data
def load_data(zip_path: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # Need to convert Year variable to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["YearInt"] = df["Year"].round().astype("Int64")  
    for c in ["Genre", "Publisher", "Platform", "Name"]:
        df[c] = df[c].astype(str).replace({"nan": None})

    # Total sales check
    df["Region_Total"] = df["NA_Sales"] + df["EU_Sales"] + df["JP_Sales"] + df["Other_Sales"]
    # (Global_Sales is already provided, but Region_Total can help sanity-check)
    return df

df = load_data(ZIP_PATH, CSV_NAME)

st.title("🎮 Video Game Sales Explorer")
st.caption("Search by genre and/or publisher, then explore sales analytics (Global + regional).")

# Sidebar Filters
st.sidebar.header("Filters")

#searchable bar for game filtering 
name_query = st.sidebar.text_input("Search game name (contains)", value="").strip()

#sorting 
genres = sorted([g for g in df["Genre"].dropna().unique() if g != "None"])
publishers = sorted([p for p in df["Publisher"].dropna().unique() if p != "None"])
platforms = sorted([p for p in df["Platform"].dropna().unique() if p != "None"])

#making sidebars
sel_genres = st.sidebar.multiselect("Game type (Genre)", options=genres, default=[])
sel_publishers = st.sidebar.multiselect("Company (Publisher)", options=publishers, default=[])
sel_platforms = st.sidebar.multiselect("Platform", options=platforms, default=[])

year_min = int(df["YearInt"].dropna().min())
year_max = int(df["YearInt"].dropna().max())
sel_years = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))

top_n = st.sidebar.slider("Top N (charts/tables)", min_value=5, max_value=30, value=10, step=1)

# Applying Filters 
f = df.copy()

if name_query:
    f = f[f["Name"].str.contains(name_query, case=False, na=False)]

if sel_genres:
    f = f[f["Genre"].isin(sel_genres)]

if sel_publishers:
    f = f[f["Publisher"].isin(sel_publishers)]

if sel_platforms:
    f = f[f["Platform"].isin(sel_platforms)]

f = f[f["YearInt"].between(sel_years[0], sel_years[1], inclusive="both")]

# Key Performance Indicators
left, mid, right, far = st.columns(4)

total_global = float(f["Global_Sales"].sum()) if len(f) else 0.0
num_titles = int(len(f))
avg_global = float(f["Global_Sales"].mean()) if len(f) else 0.0

top_game = None
top_game_sales = 0.0
if len(f):
    row = f.loc[f["Global_Sales"].idxmax()]
    top_game = row["Name"]
    top_game_sales = float(row["Global_Sales"])

left.metric("Total Global Sales (M units)", f"{total_global:,.2f}")
mid.metric("# Titles", f"{num_titles:,}")
right.metric("Avg Sales / Title (Millions)", f"{avg_global:,.2f}")
far.metric("Top Game (Global)", f"{top_game_sales:,.2f} M" if top_game else "—", top_game if top_game else "")

st.divider()

# Basic Charts: Sales by Year + Region Mix Total
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Global Sales Over Time")
   #aggregate by year
    by_year = (
        f.dropna(subset=["YearInt"])
         .groupby("YearInt", as_index=False)["Global_Sales"].sum()
         .sort_values("YearInt")
    )
    #dont plot unless there's data 
    if len(by_year):
        chart = (
            alt.Chart(by_year)
            .mark_line(point=True)
            .encode(
                x=alt.X("YearInt:O", title="Year"),
                y=alt.Y("Global_Sales:Q", title="Global Sales (M units)"),
                tooltip=["YearInt", alt.Tooltip("Global_Sales:Q", format=",.2f")]
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

with c2:
    st.subheader("Regional Mix (Total)")
    region_totals = pd.DataFrame({
        "Region": ["NA", "EU", "JP", "Other"],
        "Sales": [
            float(f["NA_Sales"].sum()),
            float(f["EU_Sales"].sum()),
            float(f["JP_Sales"].sum()),
            float(f["Other_Sales"].sum()),
        ]
    })
    # Don't plot regions without any sales
    region_totals = region_totals[region_totals["Sales"] > 0]

    if len(region_totals):
        mix = (
            alt.Chart(region_totals)
            .mark_bar()
            .encode(
                x=alt.X("Sales:Q", title="Sales (M units)"),
                y=alt.Y("Region:N", sort="-x"),
                tooltip=["Region", alt.Tooltip("Sales:Q", format=",.2f")]
            )
            .properties(height=320)
        )
        st.altair_chart(mix, use_container_width=True)
    else:
        st.info("No regional sales to display for the selected filters.")

st.divider()

# Top Publishers + Top Genres Charts 
c3, c4 = st.columns(2)

with c3:
    st.subheader("Top Publishers (by Global Sales)")
    by_pub = (
        f.groupby("Publisher", as_index=False)["Global_Sales"].sum()
         .sort_values("Global_Sales", ascending=False)
         .head(top_n)
    )
    by_pub = by_pub[by_pub["Publisher"].notna()]

    if len(by_pub):
        pub_chart = (
            alt.Chart(by_pub)
            .mark_bar()
            .encode(
                x=alt.X("Global_Sales:Q", title="Global Sales (M units)"),
                y=alt.Y("Publisher:N", sort="-x", title="Publisher"),
                tooltip=["Publisher", alt.Tooltip("Global_Sales:Q", format=",.2f")]
            )
            .properties(height=360)
        )
        st.altair_chart(pub_chart, use_container_width=True)
    else:
        st.info("No publisher data to display.")

with c4:
    st.subheader("Top Genres (by Global Sales)")
    by_genre = (
        f.groupby("Genre", as_index=False)["Global_Sales"].sum()
         .sort_values("Global_Sales", ascending=False)
         .head(top_n)
    )
    by_genre = by_genre[by_genre["Genre"].notna()]

    if len(by_genre):
        genre_chart = (
            alt.Chart(by_genre)
            .mark_bar()
            .encode(
                x=alt.X("Global_Sales:Q", title="Global Sales (M units)"),
                y=alt.Y("Genre:N", sort="-x", title="Genre"),
                tooltip=["Genre", alt.Tooltip("Global_Sales:Q", format=",.2f")]
            )
            .properties(height=360)
        )
        st.altair_chart(genre_chart, use_container_width=True)
    else:
        st.info("No genre data to display.")

st.divider()

#  Top games table
st.subheader(f"Top Games (Global Sales) — showing top {top_n}")
cols = ["Rank", "Name", "Platform", "YearInt", "Genre", "Publisher",
        "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]

top_games = (
    f.sort_values("Global_Sales", ascending=False)
     .loc[:, cols]
     .head(top_n)
)

st.dataframe(top_games, use_container_width=True, hide_index=True)

