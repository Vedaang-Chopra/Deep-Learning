import json
import os
import requests


def download_and_load_file(data_dir, file_name, url):
    file_path = data_dir / file_name
    
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data





