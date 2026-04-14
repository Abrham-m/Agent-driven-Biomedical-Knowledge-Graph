

import json
import requests

with open('dataset/papers.json', 'r') as f:
    papers = json.load(f)['papers']

for paper in papers:
    url = paper['pdf_url']
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        print(f'{url} - {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'{url} - Error: {e}')
