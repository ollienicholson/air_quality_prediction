import time 

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from helpers.helpers import save_file

# Function to fetch data from OpenAQ API
def fetch_city_data(
        city: str, 
        start_date: str, 
        end_date: str, 
        limit: int, 
        retries: int,
        dir_name: str
        ) -> pd.DataFrame | None:
    '''
    Fetches air quality data by city from OpenAQ API.
    - Set params: 
        - City: array[string]
        - Start date / End date: datetime
        - number of retries: int
    - Get results
    - Returns a Pandas DataFrame
    API docs: https://docs.openaq.org/docs/introduction
    '''
    # print("Fetching city data...")
    url = f'https://api.openaq.org/v2/measurements'
    params = {
        'city': city,
        'date_from': start_date,
        'date_to': end_date,
        'limit': limit,
        'retries': retries
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)  # Adjust timeout (seconds) as needed
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()['results']
            df = pd.DataFrame(data)
            
            filename = f'{dir_name}/{city}_data.csv'
            df.to_csv(filename) # Output the raw data to csv
            
            return df

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                raise # re-raises last exception if all retries fail



# **Cleaning data**

def clean_city_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Cleans DataFrame containing air quality measurement data.
    
    See below for more info:
    
    - Select relevant columns for visualisations, training and prediction model 
        - date: AEST, 
        - value: float, 
        - unit: µg/m³ (micrograms per cubic meter)
        - parameter: PM25
    - Normalize 'date' column
    - Convert dates and assign
    - Pivot table pivots the DataFrame data so that:
        - Each unique 'date' becomes an index - AEST
        - Each unique 'parameter' becomes a column - PM2.5
        - The 'value' associated with each combination of 'date' and 'parameter' is placed in the corresponding cell of the pivoted DataFrame.
    - Resample to daily averages
    - Drop NaN values
    - Returns a DataFrame
    '''
    try:
        # print("Cleaning city data...")
        
        data = data[['date', 'value', 'unit', 'parameter']]
        
        # Normalize the nested 'date' dictionary to extract 'utc' values
        date_utc = pd.json_normalize(data['date'])['utc']

        # Convert the extracted 'utc' dates to pandas datetime objects and assign using .loc
        data.loc[:, 'date'] = pd.to_datetime(date_utc)
        data = data.pivot_table(index='date', columns='parameter', values='value')
        
        # Resample to daily averages
        data = data.resample('D').mean()
        
        data = data.dropna()

        return data
    
    except Exception as e:
        print(f"ERROR: Could not clean data: {e}")


# **Basic Visuals**

def plot_city_data(data: pd.DataFrame, city:str, dir_name: str) -> None:
    '''
    Generate and save city data plot
    '''
    try:
        # print("Plotting city data...")
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data)
        plt.title('Daily Average AQI Levels')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.tight_layout()

        # pass directory name, function_name & city to save file
        file_name = 'aqi_daily_averages'
        save_file(dir_name, file_name, city)

    except Exception as e:
        print(f"plot_city_data ERROR {e}")

# **Training Random Fores tRegressor model**

def train_RandomForestRegressor_model(data: pd.DataFrame):
    '''
    Trains Random Forest Regressor model to predict PM2.5 levels based on day of the year.
    
    Args:
    - data: DataFrame containing time-series data with 'pm25' values and a datetime index.
    
    Returns: model 
    Trained RandomForestRegressor model.
    
    Prints:
    - Mean Squared Error (MSE) of the model on test data.
    
    Note:
    - The 'day_of_year' feature is extracted from the datetime index for modeling.
    '''
    
    try:
        # print("Training Random Forest Regressor model for city data...")
        
        data['day_of_year'] = data.index.dayofyear
        X = data[['day_of_year']]
        y = data['pm25']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test, y_pred), 3)
        logging.info(f'Mean Squared Error: {mse}')

        return model
    
    except Exception as e:
        print(f"ERROR: {e}")



def city_predict_and_plot(model, data: pd.DataFrame, city: str, dir_name: str, days: int) -> None:
    '''
    Performs predictions on future air quality index
    '''
    try:
        # print("Running prediction and plot for city data...")
        
        future_dates = pd.date_range(start=data.index[-1], periods=days)
        future_data = pd.DataFrame(
            {'day_of_year': future_dates.dayofyear}, index=future_dates)
        predictions = model.predict(future_data)

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=future_dates, y=predictions, label='Predicted AQI')
        sns.lineplot(data=data['pm25'], label='Historical AQI')
        plt.title(f"{city} AQI Prediction for next 360 Days")
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.tight_layout()
        plt.legend()
        
        file_name = 'aqi_prediction'
        # pass directory name, file_name & city to save file
        save_file(dir_name, file_name, city)
        
    except Exception as e:
        print(f"city_predict_and_plot ERROR: {e}")