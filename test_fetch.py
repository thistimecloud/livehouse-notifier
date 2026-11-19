import datetime
from main import TARGETS, fetch_website_text

today = datetime.date.today()
venues_to_check = ["Three", "近松", "近道", "BASEMENTBAR", "新宿Marble"]

for venue in venues_to_check:
    url = TARGETS[venue]
    print(f"--- {venue} ---")
    try:
        text = fetch_website_text(url, venue, today)
        if not text:
            print("Fetch failed (returned None or empty).")
        else:
            print(f"Fetched {len(text)} chars.")
    except Exception as e:
        print(f"Exception: {e}")
