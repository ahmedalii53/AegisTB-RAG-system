import os
import time
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY', '').strip('\'" ')

from google import genai
client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

for m in models_to_test:
    try:
        res = client.models.generate_content(
            model=m,
            contents="Say 'OK' in 1 word."
        )
        print(f"Model {m}: SUCCESS -> {res.text.strip()}")
    except Exception as e:
        print(f"Model {m}: FAILED -> {e}")
