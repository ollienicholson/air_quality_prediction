# import requests

# url = "https://api.openaq.org/v2/averages?locations_id=70084&spatial=location&limit=100&page=1"

# headers = {"accept": "application/json"}

# response = requests.get(url, headers=headers)

# print(response.text)

# import json

import requests

url = (
    "https://api.openaq.org/v2/cities?limit=100&page=1&offset=0&sort=asc&order_by=city"
)

headers = {"accept": "application/json"}

params = {"country": "JP", "limit": "100"}

response = requests.get(url, headers=headers, params=params)

json_response = response.json()

for items in json_response["results"]:
    print(items["country"])
