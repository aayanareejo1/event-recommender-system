from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://devpost.com/hackathons", timeout=30000)
    page.wait_for_timeout(3000)

    for i in range(15):
        page.keyboard.press("End")
        page.wait_for_timeout(2000)
        print(f"Scroll {i+1} done")

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    print(f"\nTotal text length: {len(text)}")

    input("Press Enter to close browser...")
    browser.close()