import os
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("API Key not found in environment!")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

try:
    print("Testing gemini-2.0-flash...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Hello")
    print("Success! Response:", response.text)
except Exception as e:
    print(f"Error with 2.0-flash: {e}")
    try:
        print("\nTesting gemini-1.5-flash-8b...")
        model2 = genai.GenerativeModel('gemini-1.5-flash-8b')
        response2 = model2.generate_content("Hello")
        print("Success! Response:", response2.text)
    except Exception as e2:
        print(f"Error with 1.5-flash-8b: {e2}")
