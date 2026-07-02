# Event Recommender System

A locally-run system that scrapes hackathon listings, scores them for personal relevance using a fine-tuned TensorFlow classifier, filters by location, and displays a ranked recommendation list in an interactive dashboard.

## What it does and why I built it

I was manually checking Devpost and MLH every week to find hackathons worth applying to, most of which weren't a fit for my interests or location. This project automates that: it scrapes listings, learns what I actually care about from my own labeled examples, filters out events I can't attend, and shows me a ranked shortlist instead.

## Architecture

The system is built in five independent layers, so each piece can be modified or replaced without touching the others:

1. **Data Collection** — A Playwright-driven scraper loads Devpost's hackathon page, scrolls to trigger lazy-loaded content, and uses a locally-run Gemma model (via Ollama) to extract structured event data (title, date, location, format, description) from the rendered page text.
2. **Labeling** — Scraped events are manually labeled relevant (1) or not relevant (0) based on my own judgment, producing a training dataset.
3. **ML Classifier** — A TensorFlow/Keras model scores each event's text for relevance, using pretrained sentence embeddings.
4. **Location Filter** — A separate, rule-based `LocationFilter` class checks each event's location against my current availability (Toronto, Jeddah, Online). It never sees or uses the model's output, and the model never sees location data, keeping the two layers fully independent.
5. **Output** — A Streamlit dashboard combines the model's relevance score with the location filter's pass/fail result into a final ranked, filterable list.

## Tools used and why

- **Playwright** — Devpost's hackathon listing is JavaScript-rendered with infinite scroll, so a plain HTTP request only returns an empty page shell. Playwright actually renders the page and scrolls it like a real browser would.
- **Ollama + Gemma** — Runs the extraction LLM entirely on local hardware, at zero cost, with no API keys or usage limits.
- **TensorFlow / Keras** — For the relevance classifier.
- **TensorFlow Hub's Universal Sentence Encoder** — My first model, trained with embeddings learned from scratch, failed to learn anything beyond guessing the majority class (roughly 75% accuracy, frozen across every epoch). With only ~340 labeled examples, there wasn't enough data to train useful embeddings from zero. Switching to a pretrained sentence encoder immediately produced real, improving validation accuracy (up to ~80%), since it already understands general language patterns and needed far less data to specialize.
- **Streamlit** — For the output dashboard. Note: Streamlit hangs indefinitely if it tries to load the TensorFlow Hub model live inside its process (a threading/mutex conflict on macOS). The dashboard instead reads a pre-computed CSV of ranked events, rather than running inference live.

## Requirements and design process

Full requirements, user stories, out-of-scope items, and design principles are documented in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Testing strategy

- Every scraped record is validated for required fields before being saved; malformed records are logged and skipped rather than crashing the pipeline.
- The model is evaluated on a held-out validation split every epoch, with overfitting explicitly checked for and flagged.
- The `LocationFilter` was unit tested standalone against known inputs before being integrated into the ranking pipeline.
- The full pipeline was manually verified end-to-end against real scraped data before being considered complete.

## How to run

```bash
git clone https://github.com/aayanareejo1/event-recommender-system
cd event-recommender-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install

# Scrape events
python3 src/scraper.py

# Label events (interactive terminal prompt)
python3 src/label.py

# Train the model
python3 src/train_model_use.py

# Generate ranked recommendations
python3 src/rank_events.py

# Launch the dashboard
streamlit run src/app.py
```

## Known limitations

- Data is currently sourced only from Devpost's hackathon listing; MLH and other event types (conferences, meetups) are a planned extension.
- The scraper occasionally mislabels location/format fields due to text bleeding between adjacent listings during LLM extraction; this is logged and tolerated rather than blocking the pipeline, per the project's non-functional requirements.
- The dashboard reads pre-computed scores rather than running live inference, due to a TensorFlow Hub / Streamlit threading conflict on macOS.