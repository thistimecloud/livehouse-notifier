import os
import time
import requests
import datetime
import smtplib
import json
import ast
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
import google.generativeai as genai

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# Target Live Houses
TARGETS = {
    "WWW/X": "https://www-shibuya.jp/schedule/",
    "O-Crest": "https://shibuya-o.com/crest/schedule/",
    "TOKIO TOKYO": "https://tokio.world/schedule",
    "ERA": "http://s-era.jp/schedule",
    "Three": "https://www.toos.co.jp/3/",
    "近松": "https://chikamatsu-nite.com/schedule/",
    "近道": "https://chikamichi-otemae.com/chikamichi/",
    "mona records": "https://www.mona-records.com/",
    "FEVER": "https://www.fever-popo.com/schedule/",
    "BASEMENTBAR": "https://toos.co.jp/basementbar/",
    "SHELTER": "https://www.loft-prj.co.jp/schedule/shelter/",
    "Nine Spices": "https://9spices.rinky.info/schedule/",
    "吉祥寺WARP": "http://warp.rinky.info/schedules",
    "渋谷クラブクアトロ": "https://www.club-quattro.com/shibuya/",
    "恵比寿リキッドルーム": "https://www.liquidroom.net/schedule",
}

def setup_gemini():
    if not GEMINI_API_KEY:
         raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=GEMINI_API_KEY)

def fetch_website_text(url: str, venue_name: str, target_date: datetime.date) -> str:
    """Fetches text content from a URL via BeautifulSoup."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Handle dates in URL if necessary
    target_url = url
    if "chikamatsu" in target_url or "chikamichi" in target_url:
        target_url = f"{url}{target_date.year:04d}/{target_date.month:02d}/"
    elif venue_name in ["Three", "BASEMENTBAR"]:
        # TOOS group uses dynamic paths for their event calendars: e.g., https://www.toos.co.jp/3/events/event/on/2026/02/
        base = "events/event/on/" if venue_name == "Three" else "event/on/"
        target_url = f"{url}{base}{target_date.year:04d}/{target_date.month:02d}/"
    elif venue_name == "FEVER":
        target_url = f"{url}{target_date.year:04d}/{target_date.month:02d}/"
        
    try:
        response = requests.get(target_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # TOKIO TOKYO is a Nuxt app; its text is empty, the data is in script tags.
        if venue_name != "TOKIO TOKYO":
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)
        else:
            # For TOKIO TOKYO, extract everything (or at least the JS payload)
            text = str(soup)
            
        # Limit text size to prevent exceeding token limits
        return text[:15000]
    except Exception as e:
        print(f"Error fetching {target_url}: {e}")
        return None

def extract_schedule_with_gemini(venue_name: str, text_content: str, target_date: datetime.date) -> dict:
    """Uses Gemini API to extract details from the webpage text."""
    if not text_content:
        return {"error": "取得失敗 (テキストなし)"}
        
    date_str_long = target_date.strftime("%Y年%m月%d日")
    date_str_short = target_date.strftime("%m/%d")
    date_str_short2 = f"{target_date.month}.{target_date.day}"
    dd = str(target_date.day)

    prompt = f"""
以下はライブハウス「{venue_name}」のスケジュールページのテキストデータです。
この中から、指定された日付（{date_str_long} または {date_str_short} または {date_str_short2} または単なる「{dd}日」など）に行われるライブ・イベントの情報をすべて抽出してください。

テキストには複数の日付の情報が含まれていますが、必ず指定された日の情報だけを抜き出してください。
同じ日に複数のイベントがある場合は、すべて含めてください。
もし当日にイベントが全く見つからない場合は、必ず [{{
"has_live": false}}] を返してください。
（※"詳細未定"や"TBA"のような表記でも、何かが記載されていれば"has_live": trueとして抽出してください）

抽出結果は、必ず以下のJSON配列フォーマットのみ（マークダウンのコードブロックは不要、角括弧から始めること）で返してください。イベントが1つの場合も必ず配列にしてください:
[
  {{
    "has_live": true,
    "title": "イベントのタイトル",
    "artists": ["アーティスト名1", "アーティスト名2"],
    "open_start": "開場 / 開演時間",
    "adv_door": "前売 / 当日などのチケット料金",
    "remarks": "その他の特記事項（あれば）"
  }}
]

--- テキストデータ ---
{text_content}
"""

    # Using gemini-2.0-flash as the best general capability model. 
    # If the user hits Limit: 0 on 2.0-flash, we will try 1.5-flash-8b.
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"Request failed with gemini-2.5-flash: {e}")
        print("Falling back to gemini-flash-latest...")
        try:
            model = genai.GenerativeModel(
                model_name="gemini-flash-latest",
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            response = model.generate_content(prompt)
        except Exception as e2:
             print(f"Gemini API Error for {venue_name}: {e2}")
             return {"error": "Gemini API エラー (課金・リミット制限の可能性)"}
        
    try:
        result_text = response.text.strip()
        
        # Extract JSON array first, then fallback to object
        start_idx = result_text.find('[')
        end_idx = result_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            json_str = result_text[start_idx:end_idx+1]
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = json_str.replace("true", "True").replace("false", "False")
                parsed = ast.literal_eval(json_str)
            # Ensure always a list
            return parsed if isinstance(parsed, list) else [parsed]
        else:
            # Fallback: try to parse as a single object
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = result_text[start_idx:end_idx+1]
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = json_str.replace("true", "True").replace("false", "False")
                    parsed = ast.literal_eval(json_str)
                return [parsed]
            return [{"error": "AI出力からJSON形式が見つかりませんでした"}]
            
    except Exception as e:
        print(f"Parsing error for {venue_name}: {e}")
        return [{"error": "AI結果の解析に失敗しました"}]

# --- Formatting & Notification ---

def format_message(results: dict, target_date: datetime.date, url_map: dict = None) -> str:
    """Formats the extracted results into a plain text message string matching previous nice format."""
    date_str = target_date.strftime("%m/%d(%a)")
    message = f"🎸 本日 {date_str} のライブ情報 🎸\n"
    message += "=" * 30 + "\n"
    
    for venue, events in results.items():
        url = url_map.get(venue, "") if url_map else ""
        if url:
            message += f"\n📍 【[{venue}]({url})】\n"
        else:
            message += f"\n📍 【{venue}】\n"
        # Normalize to list (supports both old dict format and new list format)
        if isinstance(events, dict):
            events = [events]
        
        live_events = [e for e in events if isinstance(e, dict) and e.get("has_live")]
        errors = [e for e in events if isinstance(e, dict) and "error" in e]
        
        if errors:
            message += f"⚠️ 情報取得エラー: {errors[0].get('error', 'Unknown Error')}\n"
        elif not live_events:
            message += "❌ 公演なし / 予定なし\n"
        else:
            for i, info in enumerate(live_events):
                if len(live_events) > 1:
                    message += f"--- 公演 {i+1} ---\n"
                if info.get("title"): message += f"🏷️ {info['title']}\n"
                if info.get("artists"): message += f"🎤 {', '.join(info['artists'])}\n"
                if info.get("open_start"): message += f"⌚ {info['open_start']}\n"
                if info.get("adv_door"): message += f"💴 {info['adv_door']}\n"
                if info.get("remarks"): message += f"📝 {info['remarks']}\n"
        message += "-" * 30 + "\n"
        
    return message

def send_email(subject: str, body: str):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        print("Email configuration is incomplete. Skipping.")
        return
    # カンマ区切りで複数の受信者に対応
    receivers = [r.strip() for r in EMAIL_RECEIVER.split(',')]
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = ', '.join(receivers)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, receivers, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {len(receivers)} recipient(s).")
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_discord_webhook(message: str):
    if not DISCORD_WEBHOOK_URL:
        print("Discord Webhook URL is not set. Skipping.")
        return
    if len(message) > 2000:
        message = message[:1990] + "...(省略)"
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
        response.raise_for_status()
        print("Discord notification sent successfully.")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def main():
    try:
        setup_gemini()
    except Exception as e:
        print(e)
        return
        
    today = datetime.date.today()
    print(f"Running scraper for {today}...")
    
    all_results = {}
    
    for venue_name, url in TARGETS.items():
        print(f"Fetching data for {venue_name}...")
        text_content = fetch_website_text(url, venue_name, today)
        
        # TOKIO TOKYO fallback for blank blocking pages or failed requests
        if not text_content or len(text_content) < 50:
            all_results[venue_name] = {"error": "取得失敗 または ブロック回避"}
            continue
            
        print(f"AI parsing for {venue_name}...")
        info = extract_schedule_with_gemini(venue_name, text_content, today)
        all_results[venue_name] = info
        
        # Crucial delay to prevent hitting free tier rate limits (15 TPM on free models)
        print("Waiting 15 seconds to respect API Rate Limits...")
        time.sleep(15)
            
    print("\nFormatting message...")
    final_message = format_message(all_results, today, url_map=TARGETS)
    
    print("\n--- Message Preview ---")
    print(final_message)
    print("-----------------------\n")
    
    print("Sending notifications...")
    date_str = today.strftime("%Y/%m/%d")
    subject = f"🎸 本日のライブ情報 ({date_str})"
    
    send_email(subject, final_message)
    send_discord_webhook(final_message)
    
    print("Done!")

if __name__ == "__main__":
    main()
