import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-6392c92de0247364ef14d229e0b4ceb2730ee661a674a84c"
}

# Prompt longo para tentar disparar o otimizador
long_text = "Repita a palavra 'TESTE' 500 vezes para criar um volume de tokens. " + "TESTE " * 500
payload = {
    "model": "gpt-5.5",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Analise o seguinte texto e resuma: {long_text}"}
    ]
}

print("Enviando requisição longa...")
try:
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Sucesso! Verifique agora os logs do manager.")
    else:
        print(f"Erro: {response.text}")
except Exception as e:
    print(f"Erro na requisição: {e}")
