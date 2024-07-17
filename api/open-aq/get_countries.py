import json

import requests

# Gets countries

url = "https://api.openaq.org/v3/countries?order_by=id&sort_order=asc&limit=100&page=1"

headers = {"accept": "application/json"}

# params = {"limit": "200"}

response = requests.get(url, headers=headers)

# json_response = response.json()

print(response.text)

# print(json.dumps(json_response, indent=4))

# for item in json_response


# if "results" in json_response:
#     seen = set()
#     for item in json_response["results"]:
#         # Create a unique identifier for each city-country pair
#         identifier = item["country"]
#         if identifier not in seen:
#             seen.add(identifier)
#             print(f"Country: {item["country"]}")
#             # print(f"City: {item["city"]}")
#     # print(seen)
# else:
#     print("No items in json_response")
