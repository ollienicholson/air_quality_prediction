import os
from metaflow import FlowSpec, step
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import pandas as pd
import time


# Create image output directory if it doesn't exist
if not os.path.exists('image_outputs'):
    os.makedirs('image_outputs')
    

# Create csv output directory if it doesn't exist
if not os.path.exists('csv_outputs'):
    os.makedirs('csv_outputs')


# Function to fetch data from OpenAQ API
def fetch_data(
        city: str, 
        start_date: str, 
        end_date: str, 
        limit: int, 
        retries=3
        ):
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
            
            filename = f'{city}_data.csv'
            df.to_csv(filename) # Output the raw data to csv
            
            return df

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                raise  # Raise exception if all retries fail





# **Cleaning data**

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
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

def plot_data(data: pd.DataFrame, city):
    '''
    NOTE: update function description
    '''
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data)
    plt.title('Daily Average AQI Levels')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.tight_layout()
    plt.show()

    # Define the base filename
    base_filename = f"{city}_aqi_daily_averages.png"
    
    # Check if the file exists and increment the counter if needed
    counter = 1
    while os.path.exists(base_filename):
        base_filename = f"{city}_aqi_daily_averages_{counter}.png"
        counter += 1
    
    # Save the plot to the data directory
    plt.savefig(f"image_outputs/{base_filename}")


# **Training the model**

def train_model(data):
    '''
    Trains a RandomForestRegressor model to predict PM2.5 levels based on day of the year.
    
    Args:
    - data: DataFrame containing time-series data with 'pm25' values and a datetime index.
    
    Returns:
    - model: Trained RandomForestRegressor model.
    
    Prints:
    - Mean Squared Error (MSE) of the model on test data.
    
    Note:
    - The 'day_of_year' feature is extracted from the datetime index for modeling.
    '''
    
    try:
        data['day_of_year'] = data.index.dayofyear
        X = data[['day_of_year']]
        y = data['pm25']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')

        return model
    
    except Exception as e:
        print(f"ERROR: {e}")



def predict_and_plot(model, data, city):
    '''
    NOTE: update function description
    '''
    try:
        future_dates = pd.date_range(start=data.index[-1], periods=360)
        future_data = pd.DataFrame(
            {'day_of_year': future_dates.dayofyear}, index=future_dates)
        predictions = model.predict(future_data)

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=future_dates, y=predictions, label='Predicted AQI')
        sns.lineplot(data=data['pm25'], label='Historical AQI')
        plt.title('AQI Prediction for Next 360 Days')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.tight_layout()
        plt.legend()
        plt.show()

        # Save the plot to the data directory
        plt.savefig(f'image_outputs/{city}_aqi_prediction.png')
    
    except Exception as e:
        print(f"ERROR: {e}")


# Fetch data for a specific city and date range
city = 'Melbourne'
start_date = '2021-01-01'
end_date = '2023-12-31'
limit = 12000


print("Fetching data...")
data = fetch_data(city, start_date, end_date, limit)


print("Cleaning data...")
cleaned_data = clean_data(data)


print("Plotting data...")
print("Close image to continue...")
plot_data(cleaned_data, city)


print("Training model...")
model = train_model(cleaned_data)


print("Running Predict and Plot...")
predict_and_plot(model, cleaned_data, city)

print("Completed")