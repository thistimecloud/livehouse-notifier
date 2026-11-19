import requests
from bs4 import BeautifulSoup
import datetime

TARGETS = {
    "TOKIO TOKYO": "https://tokio.world/schedule",
    "Three": "https://www.toos.co.jp/3/",
    "BASEMENTBAR": "https://toos.co.jp/basementbar/"
}

def fetch_website_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        print(f"[{url}] Status: {response.status_code}")
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return text[:500] # Just look at the first 500 chars to see if it even loaded content
    except Exception as e:
        return f"Error: {e}"

for name, url in TARGETS.items():
    print(f"\n--- {name} ---")
    text = fetch_website_text(url)
    print(text)

