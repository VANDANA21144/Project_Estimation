import requests

files = {'file': open('models/rf_teamexp_classifier.pkl', 'rb')}
r = requests.post('http://127.0.0.1:8000/admin/upload-model', files=files)

print("Status:", r.status_code)
print("Response:", r.text)
