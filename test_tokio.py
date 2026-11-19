import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
}

def test_url(url):
    print(f"Testing {url}...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        print(f"Status: {r.status_code}")
        html = r.text
        # Look for script tags containing JSON data common in Next.js or Nuxt
        import re
        if '__NEXT_DATA__' in html:
            print("Found __NEXT_DATA__")
        elif 'pageProps' in html:
            print("Found pageProps")
        print(f"Total HTML length: {len(html)}")
        print(html[:500])
    except Exception as e:
        print(f"Error: {e}")

test_url("https://tokio.world/schedule")
