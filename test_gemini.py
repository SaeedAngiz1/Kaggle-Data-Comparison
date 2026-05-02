import requests
import json

llm_url = 'https://generativelanguage.googleapis.com/v1beta'
model_name = 'gemini-1.5-pro'
api_key = 'FAKE_KEY'
endpoint = f'{llm_url.rstrip("/")}/models/{model_name}:generateContent?key={api_key}'

prompt = 'test'
system_prompt = 'system'
headers = {'Content-Type': 'application/json'}

payload = {
    'contents': [{'parts': [{'text': prompt}]}]
}
if system_prompt:
    payload['systemInstruction'] = {'parts': [{'text': system_prompt}]}

try:
    response = requests.post(endpoint, json=payload, headers=headers)
    print(f'STATUS: {response.status_code}')
    print(f'RESPONSE: {response.text}')
except Exception as e:
    print(f'ERROR: {e}')
