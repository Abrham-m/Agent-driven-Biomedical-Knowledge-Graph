import os
import json
import requests

def download_papers(json_path, download_folder):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    papers = data.get('papers', [])
    os.makedirs(download_folder, exist_ok=True)
    for paper in papers:
        pdf_url = paper['pdf_url']
        title = paper['title']
        # Create a safe filename from title
        filename = title.replace(' ', '_').replace('/', '_')[:50] + '.pdf'
        filepath = os.path.join(download_folder, filename)
        print(f"Downloading: {title}\nFrom: {pdf_url}")
        try:
            headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                }
            response = requests.get(pdf_url, headers=headers, timeout=60)
            #response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Saved to: {filepath}\n")
        except Exception as e:
            print(f"Failed to download {title}: {e}\n")

if __name__ == "__main__":
    json_path = os.path.join(os.path.dirname(__file__), 'papers.json')
    download_folder = os.path.join(os.path.dirname(__file__), 'research_papers')
    download_papers(json_path, download_folder)
