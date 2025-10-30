import requests
import random

URL = "http://127.0.0.1:8000/predict/"

dummy_sequence = [[random.uniform(0, 1) for _ in range(6)] for _ in range(10)]

data = {
    "sequence": dummy_sequence
}

response = requests.post(URL, json=data)

print("✅ Response status:", response.status_code)
print("📈 Prediction:", response.json())
