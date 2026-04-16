import requests
import json

login = requests.post("http://localhost:8000/api/auth/login", data={
    "username": "ali@test.com",
    "password": "123456"
})
token = login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
session_id = "2f9a0c2b-f2b1-40da-8efb-883656abc67d"

# Get full report
report = requests.get(
    f"http://localhost:8000/api/sessions/{session_id}/report",
    headers=headers
)

# Save to file
with open("report.json", "w", encoding="utf-8") as f:
    json.dump(report.json(), f, indent=2, ensure_ascii=False)

print("Saved to report.json")
print(json.dumps(report.json(), indent=2, ensure_ascii=False))