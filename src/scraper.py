import json
import csv
import time
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests

def clean_json_text(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def get_rendered_chunks():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://devpost.com/hackathons", timeout=30000)
        page.wait_for_timeout(3000)

        for i in range(40):
            page.keyboard.press("End")
            page.wait_for_timeout(2000)
            print(f"Scroll {i+1}/40 done")

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    chunk_size = 3000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def extract_events(content, chunk_num):
    payload = {
        "model": "gemma3:4b",
        "prompt": f"""Extract all real hackathon events from this page content.
For each return: title, date, location, format, description.
Only include real hackathons, not navigation text or unrelated content.
Return ONLY a raw JSON list, no markdown, no code fences, no extra text.

Content:
{content}
""",
        "stream": False,
        "options": {
            "num_predict": 4096
        }
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    raw_text = response.json()["response"]
    cleaned = clean_json_text(raw_text)
    try:
        events = json.loads(cleaned)
        return events if isinstance(events, list) else []
    except json.JSONDecodeError:
        print(f"Could not parse chunk {chunk_num}, saving to debug log")
        with open(f"data/failed_chunk_{chunk_num}.txt", "w") as f:
            f.write(raw_text)
        return []

print("Loading and scrolling Devpost hackathons page...")
chunks = get_rendered_chunks()
print(f"\nGot {len(chunks)} content chunks to process.")

all_events = []
seen_titles = set()

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")
    events = extract_events(chunk, i)
    for e in events:
        if not e.get("title") or not e.get("date"):
            print(f"Skipping malformed record: {e}")
            continue
        title = e["title"].strip()
        if not e.get("location"):
            e["location"] = "Unknown"
        if not e.get("format"):
            e["format"] = "Unknown"
        if not e.get("description"):
            e["description"] = e.get("format", "")
        if title not in seen_titles:
            seen_titles.add(title)
            all_events.append(e)
    time.sleep(1)

print(f"\nTotal unique valid events collected: {len(all_events)}")

with open("data/raw_events.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "date", "location", "format", "description"])
    writer.writeheader()
    writer.writerows(all_events)

print("Saved to data/raw_events.csv")