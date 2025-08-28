import requests

response1 = requests.post(
    "http://localhost:8000/openai/novel/invoke", 
    json={'input': {'topic': '행복에 대해서'}}
)

response2 = requests.post(
    "http://localhost:8000/openai/poem/invoke", 
    json={'input': {'topic': '행복에 대해서'}}
)

response3 = requests.post(
    "http://localhost:8000/ollama/novel/invoke", 
    json={'input': {'topic': '행복에 대해서'}}
)

response4 = requests.post(
    "http://localhost:8000/ollama/poem/invoke", 
    json={'input': {'topic': '행복에 대해서'}}
)

print(response1.json())
print("\n")
print(response2.json())
print("\n")
print(response3.json())
print("\n")
print(response4.json())