import streamlit as st
import pandas as pd

st.set_page_config(page_title="Event Recommender", page_icon="🎯", layout="wide")

RANKED_PATH = "data/ranked_recommendations.csv"

@st.cache_data
def load_ranked():
    return pd.read_csv(RANKED_PATH)

st.title("🎯 Event Recommender System")
st.caption("Personalized hackathon and tech event recommendations, scored locally by a fine-tuned relevance model.")

df = load_ranked()

with st.sidebar:
    st.header("Filters")
    min_score = st.slider("Minimum relevance score", 0.0, 1.0, 0.3, 0.05)
    top_n = st.slider("Number of results to show", 5, 100, 20)
    location_search = st.text_input("Filter by location contains", value="")

filtered = df[df["relevance_score"] >= min_score]
if location_search.strip():
    filtered = filtered[filtered["location"].str.contains(location_search, case=False, na=False)]

filtered = filtered.sort_values("relevance_score", ascending=False).head(top_n)

st.subheader(f"Showing {len(filtered)} of {len(df)} events")

for _, row in filtered.iterrows():
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{row['title']}**")
            st.caption(f"📅 {row['date']}  •  📍 {row['location']}")
        with col2:
            st.metric("Relevance", f"{row['relevance_score']:.2f}")

if len(filtered) == 0:
    st.info("No events match your current filters. Try lowering the minimum score.")