import requests

API_URL = "https://api-inference.huggingface.co/models/hamzab/roberta-fake-news-classification"
headers = {"Authorization": "Bearer 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"}
payload = {"inputs": "NASA scientists confirmed water ice on the Moon"}

response = requests.post(API_URL, headers=headers, json=payload)
print("Status code:", response.status_code)
print("Response:", response.json())