import logging
import os

from aqi_functions_city import fetch_city_data, clean_city_data, plot_city_data, \
                                train_RandomForestRegressor_model, city_predict_and_plot


# change csv output as per user preferences
csv_output_directory = 'csv'

# Create csv output directory if it doesn't exist
if not os.path.exists(csv_output_directory):
    os.makedirs(csv_output_directory)
    
# change directory output as per user preferences
image_output_directory = 'img'

# Create image output directory if it doesn't exist
if not os.path.exists(image_output_directory):
    os.makedirs(image_output_directory)


logging.basicConfig(
    filename='/Users/olivernicholson/github/air-quality-project/logs/air-quality-project-cron.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Script started")


# Assign variables
city = 'Melbourne'
start_date = '2021-01-01'
end_date = '2023-12-31'
limit = 12000
retries = 3
num_prediction_days = 360 # in days

try:
    logging.info("Running city AQ pipeline at:", os.getcwd())
    data = fetch_city_data(city, start_date, end_date, limit, retries, csv_output_directory)
    logging.info("Fetched city data")

    cleaned_data = clean_city_data(data)
    logging.info("Cleaned city data")

    plot_city_data(cleaned_data, city, image_output_directory)
    logging.info("Plotted city data")

    model = train_RandomForestRegressor_model(cleaned_data)
    logging.info("Trained RandomForestRegressor model")

    city_predict_and_plot(model, cleaned_data, city, image_output_directory, num_prediction_days)
    logging.info("Prediction ran. Plotted predicted city data")
    
    logging.info("AQI functions for city data completed")
    
except Exception as e:
    logging.error("Error occurred: %s", str(e))