import requests

url = "http://127.0.0.1:8000/predict/"
image_path = "img-1.jpg"
files = {"file": open(image_path, "rb")}

response = requests.post(url, files=files)
print(response.json())
