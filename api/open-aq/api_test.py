import json
import time

import pandas as pd
import requests


def fetch_city_data(
    city: str, start_date: str, end_date: str, limit: int, retries: int, dir_name: str
) -> pd.DataFrame | None:
    """
    Fetches air quality data by city from OpenAQ API.
    - Set params:
        - City: array[string]
        - Start date / End date: datetime
        - number of retries: int
    - Get results
    - Returns a Pandas DataFrame
    API docs: https://docs.openaq.org/docs/introduction
    """
    # print("Fetching city data...")
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "city": city,
        "date_from": start_date,
        "date_to": end_date,
        "limit": limit,
        "retries": retries,
    }
    for attempt in range(retries):
        try:
            response = requests.get(
                url, params=params, timeout=10
            )  # Adjust timeout (seconds) as needed
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()["results"]
            df = pd.DataFrame(data)

            filename = f"{dir_name}/{city}_data.csv"
            df.to_csv(filename)  # Output the raw data to csv

            return df

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                raise  # re-raises last exception if all retries fail


url = "https://api.openaq.org/v2/measurements?date_from=2024-05-30T00%3A00%3A00Z&date_to=2024-06-06T20%3A45%3A00Z&limit=100&page=1&offset=0&sort=desc&radius=1000&order_by=datetime"

headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)
measurement_data = response.json()
measurement_file = "measurement-data.json"

with open(measurement_file, "w") as file:
    json.dump(measurement_data, file, indent=4)


header_data = dict(response.headers)
header_file = "measurement-headers.json"
with open(header_file, "w") as file:
    json.dump(header_data, file, indent=4)
