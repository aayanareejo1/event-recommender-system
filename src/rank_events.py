import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
from location_filter import LocationFilter

MODEL_PATH = "data/event_relevance_model_use.keras"
EVENTS_PATH = "data/raw_events.csv"

USER_LOCATIONS = ["Toronto", "Jeddah", "Online"]

print("Loading model...")
model = tf_keras.models.load_model(
    MODEL_PATH,
    custom_objects={"KerasLayer": hub.KerasLayer}
)

print("Loading events...")
df = pd.read_csv(EVENTS_PATH)
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

print("Scoring events with the ML model...")
scores = model.predict(df["text"].values, verbose=0).flatten()
df["relevance_score"] = scores

print("Applying location filter...")
location_filter = LocationFilter(USER_LOCATIONS)
df["location_pass"] = df["location"].apply(location_filter.passes)

ranked = df[df["location_pass"] == True].copy()
ranked = ranked.sort_values("relevance_score", ascending=False)

print(f"\n{len(df)} total events, {len(ranked)} pass the location filter\n")
print("Top 15 ranked recommendations:\n")

for i, row in ranked.head(15).iterrows():
    print(f"{row['relevance_score']:.3f}  |  {row['title']}  ({row['location']})")

ranked[["title", "date", "location", "relevance_score"]].to_csv(
    "data/ranked_recommendations.csv", index=False
)
print("\nFull ranked list saved to data/ranked_recommendations.csv")