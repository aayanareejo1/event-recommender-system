# Event Recommender System — Requirements Document

## Project Overview
A locally-run system that scrapes hackathon and tech event listings using an 
LLM-based scraper, trains a TensorFlow text classifier to score events by 
personal relevance, and filters results by location and format, producing a 
ranked recommendation list.

## User Stories

**Core**
- As a user, I want to see a ranked list of hackathons and tech events so that 
  I can decide which ones are worth applying to without manually checking 
  multiple sites.
- As a user, I want events filtered by my current location (Toronto, Jeddah, 
  or online) so that I do not see irrelevant in-person events I cannot attend.
- As a user, I want the system to learn from my past labeling so that future 
  recommendations reflect my actual interests without me re-specifying 
  preferences each time.

**Secondary**
- As a user, I want malformed or incomplete event records to be skipped 
  gracefully so that the pipeline does not crash on bad data.
- As a user, I want to be able to retrain the model on new labeled data so 
  that recommendations improve over time.

## Out of Scope
- The system does not register or apply to events on the user's behalf. Recommendation only.
- The system does not scrape sites that explicitly prohibit scraping in their terms of service.
- The system does not retrain automatically. Retraining is a deliberate manual step when new labeled data is available.
- The system does not send notifications or alerts about upcoming events.
- The system does not support multiple user profiles. It is built for a single user.

## Non-Functional Requirements
- Must run entirely at zero ongoing cost. No paid APIs anywhere in the pipeline.
- Must run locally end-to-end. No event data or personal preferences leave the user's machine.
- Must tolerate incomplete or malformed scraped records without crashing. Invalid records are logged and skipped.
- Must be reproducible. Anyone cloning the repo and following the README should be able to run the full pipeline on their own machine.

## Testing Strategy

### Data Collection Layer (EventScraper)
- Verify every scraped record contains all required fields: title, date, location, format, and description.
- Log and skip records missing any required field rather than failing the entire run.
- Run a schema validation check after every scraping session before saving to disk.

### Preprocessing Layer
- Verify TextVectorization output produces a consistent shape across all inputs regardless of text length.
- Confirm no null or empty inputs reach the vectorization step.

### ML Classifier (EventClassifier)
- Always evaluate on a held-out validation set, not just training data.
- Track both training accuracy and validation accuracy each epoch to catch overfitting early.
- If validation accuracy diverges significantly from training accuracy, flag as an overfitting issue before proceeding.

### Location Filter (LocationFilter)
- Unit test the filter with known inputs and expected outputs before integrating with the model.
- Confirm the filter operates independently from the model. The model receives no location data, and the filter accesses no model internals.

### End-to-End
- Run the full pipeline on a small known sample of 10 to 20 labeled events and manually verify the final ranked output looks reasonable before running on the full dataset.

## Design Principles Applied

**Separation of Concerns**
The ML scoring layer and the rule-based filtering layer are kept fully independent. The classifier scores events based on text content only. The location filter applies location and format rules after scoring. Either layer can be modified or replaced without touching the other.

**Graceful Degradation**
The pipeline never crashes on bad data. Malformed records are logged and skipped so the rest of the run completes successfully.

**Zero Cost by Design**
Every tool in the stack runs locally or on a free tier. The LLM backbone (Gemma 4 via Ollama) runs on local hardware. No API keys are required to run the project.