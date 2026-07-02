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
## Technical deep dive: problems hit and how they were solved

**Scraping a JavaScript-rendered, infinite-scroll page**
Devpost's hackathon listing loads content dynamically; a plain `requests.get()` call only returned the empty page shell. Switched to Playwright to actually render the page in a headless browser. Standard `mouse.wheel()` scrolling didn't trigger the lazy-loading; `page.keyboard.press("End")` did. Raw HTML (including embedded `<script>` tags) confused the extraction LLM into hallucinating structure from JavaScript, fixed by stripping `script`/`style` tags with BeautifulSoup before passing text to the model. Large content chunks caused the LLM's JSON output to get truncated mid-object; fixed by reducing chunk size to 3000 characters and raising `num_predict` to 4096. Final result: 345 unique, real hackathon events collected and deduplicated by title.

**Baseline ML model failure and diagnosis**
The first classifier (TextVectorization + embeddings trained from scratch) showed validation accuracy frozen at exactly ~75% across all 20 training epochs, a sign it was just predicting the majority class rather than learning. With only ~340 labeled examples, there wasn't enough data to train useful word embeddings from zero. Diagnosed by noticing validation accuracy never varied between epochs even as training accuracy fluctuated. Fixed by swapping to TensorFlow Hub's pretrained Universal Sentence Encoder (transfer learning), which already understands general language and needed far less data to specialize. Validation accuracy then climbed steadily to ~80% with real epoch-over-epoch improvement.

**Separation of concerns**
The `LocationFilter` class was built and unit tested completely independently of the ML model, per the project's design principles, the model never receives location data, and the filter never accesses model internals. This means either layer can be modified or replaced without touching the other.

**Streamlit + TensorFlow Hub deadlock**
Loading the TensorFlow Hub model live inside Streamlit's process caused an indefinite hang (a low-level mutex lock conflict specific to how Streamlit runs scripts on macOS). Worked around by decoupling inference from the UI: rankings are precomputed to a CSV by a separate script, and the Streamlit dashboard only reads and filters that CSV, no live model loading inside the web app.

**Git history and large files**
Accidentally committed the full `venv/` folder and an 980MB saved model file (the full Universal Sentence Encoder gets serialized into `.keras` files by default), exceeding GitHub's 100MB file limit and blocking pushes. Fixed with a proper `.gitignore` (excluding `venv/` and `*.keras`) and a full git history reset to remove the large files from prior commits entirely.