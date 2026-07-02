import requests
import json

# Step 1: fetch raw HTML
url = "https://devpost.com/hackathons"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text[:8000]

# Step 2: send to local Ollama
payload = {
    "model": "gemma3:4b",
    "prompt": f"""Extract up to 3 hackathon events from this HTML. 
For each event return: title, date, location, format, description.
Return as a JSON list only, no extra text.

HTML:
{html}
""",
    "stream": False
}

response = requests.post("http://localhost:11434/api/generate", json=payload)
result = response.json()
print(result["response"])