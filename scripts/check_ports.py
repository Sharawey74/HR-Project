
import requests

ports = [11434, 11500]
print("=== Ollama Connection Check ===")
for port in ports:
    url = f"http://localhost:{port}/api/tags"
    try:
        print(f"Testing port {port}...", end=" ")
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            print(f"SUCCESS! (Server found)")
        else:
            print(f"Connected but returned {resp.status_code}")
    except Exception as e:
        print(f"FAILED (Connection refused)")
