import google.generativeai as genai

# !!! PASTE YOUR API KEY HERE !!!
API_KEY = "AIzaSyDdcMkKuO6VUngqlBA9dIjDiKZD0JePoJ8"

genai.configure(api_key=API_KEY)

print("--- Listing Available Models ---")
try:
    for m in genai.list_models():
        # Only show models that support text generation
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")