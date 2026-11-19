import requests
from bs4 import BeautifulSoup

url = "https://tokio.world/schedule"
headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.content, "html.parser")
for script in soup(["script", "style"]):
    script.decompose()
text = soup.get_text(separator='\n', strip=True)
print(f"Text length: {len(text)}")
print(text[:1000])
