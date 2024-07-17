# Gets cities

import requests

url = (
    "https://api.openaq.org/v2/cities?limit=100&page=1&offset=0&sort=asc&order_by=city"
)

headers = {"accept": "application/json"}

params = {"limit": "200"}

response = requests.get(url, headers=headers, params=params)

json_response = response.json()

if "results" in json_response:
    seen = set()
    for item in json_response["results"]:
        # Create a unique identifier for each city-country pair
        identifier = item["country"]
        if identifier not in seen:
            seen.add(identifier)
            print(f"Country: {item["country"]}")
            # print(f"City: {item["city"]}")
    # print(seen)
else:
    print("No items in json_response")
