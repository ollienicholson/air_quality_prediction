import os
import logging

from aqi_functions_city import fetch_city_data, clean_city_data, plot_city_data, train_RandomForestRegressor_model, city_predict_and_plot

# Create image output directory if it doesn't exist
if not os.path.exists('image_outputs'):
    os.makedirs('image_outputs')
    

# Create csv output directory if it doesn't exist
if not os.path.exists('csv_outputs'):
    os.makedirs('csv_outputs')


# Fetch data for a specific city and date range
city = 'Melbourne'
start_date = '2021-01-01'
end_date = '2023-12-31'
limit = 12000

data = fetch_city_data(city, start_date, end_date, limit)

cleaned_data = clean_city_data(data)

plot_city_data(cleaned_data, city)

model = train_RandomForestRegressor_model(cleaned_data)

city_predict_and_plot(model, cleaned_data, city)

print("AQI functions for city data completed")