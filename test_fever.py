import requests

url = "https://www.fever-popo.com/schedule/"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
}
try:
    r = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
