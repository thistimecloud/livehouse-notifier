import re
import datetime
import requests
from bs4 import BeautifulSoup

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
         r = requests.get(url, headers=headers, timeout=10)
         if r.status_code == 200:
             return BeautifulSoup(r.content, 'html.parser')
    except:
         pass
    return None

TARGETS = {
    "WWW/X": "https://www-shibuya.jp/schedule/",
    "O-Crest": "https://shibuya-o.com/crest/schedule/",
    "ERA": "http://s-era.jp/schedule",
    "Three": "https://www.toos.co.jp/3/",
    "近松": "https://chikamatsu-nite.com/schedule/",
    "近道": "https://chikamichi-otemae.com/chikamichi/",
    "mona records": "https://www.mona-records.com/",
    "FEVER": "https://www.fever-popo.com/",
    "BASEMENTBAR": "https://toos.co.jp/basementbar/",
    "SHELTER": "https://www.loft-prj.co.jp/schedule/shelter/",
    "Nine Spices": "https://9spices.rinky.info/schedule/"
}
target_date = datetime.date(2026, 2, 25)

# More robust regex patterns for finding the date
patterns = [
    re.compile(r'2026[./年]0?2[./月]25日?'),
    re.compile(r'(?<!\d)0?2[./月]25日?'),
    # For calendars that just have "25" we need to be careful, but we can look for specific class names containing the day
]

for name, url in TARGETS.items():
    if "chikamatsu" in url or "chikamichi" in url:
        url += "2026/02/"
    soup = get_soup(url)
    if not soup:
        print(f"[{name}] Failed")
        continue
    
    # 1. search common date classes explicitly
    found = False
    for el in soup.find_all(class_=re.compile(r'date|day|time|sch', re.I)):
        text = el.get_text(strip=True)
        if any(p.search(text) for p in patterns) or text == "25" or text == "25日":
            print(f"[{name}] Found explicitly in class: {text}")
            found = True
            break
            
    if not found:
        # 2. search anywhere
        text = soup.get_text()
        if any(p.search(text) for p in patterns):
             print(f"[{name}] Found in raw text")
        else:
             print(f"[{name}] NOT FOUND")
