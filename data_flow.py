from metaflow import FlowSpec, step

from aqi_functions_city import fetch_city_data, clean_city_data, \
                                train_RandomForestRegressor_model, city_predict_and_plot

# to run this flow:
# python3 data_flow.py run
class AQIPredictionFlow(FlowSpec):
    '''
    This Metaflow class defines an end-to-end pipeline for predicting Air Quality Index (AQI) levels for a specified city. This pipeline includes steps for data fetching, cleaning, model training, and generating future AQI predictions along with visualizations.

    Steps in pipeline:
    
    1. `start`: Initialize the flow by setting up parameters such as city name, date range for historical data, and the number of days to predict into the future.

    2. `fetch_data`: Fetch historical AQI data for the specified city and date range.

    3. `clean_data`: Clean and preprocess the fetched AQI data to prepare it for model.

    4. `train_model`: Train a Random Forest Regressor model using the cleaned data.

    5. `predict_and_plot`: Use the trained model to predict future AQI levels for the specified number of days.
       Generate and display a plot of the historical and predicted AQI levels.

    6. `end`: Completion of the AQI prediction pipeline.
    '''

    @step
    def start(self):
        
        print("Starting data flow...")
        
        import os
        
        self.city = 'Melbourne'
        self.start_date = '2021-01-01'
        self.end_date = '2023-12-31'
        self.limit = 12000
        self.retries = 3
        self.num_prediction_days = 360
        
        # change csv output as per user preferences
        self.csv_dir = 'csv'

        # Create csv output directory if it doesn't exist
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
            
        # change directory output as per user preferences
        self.img_dir = 'img'

        # Create image output directory if it doesn't exist
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
    
        self.next(self.fetch_data)

    @step
    def fetch_data(self):
        print("Fetching data...")
        self.dataframe = fetch_city_data(
            self.city, 
            self.start_date, 
            self.end_date, 
            self.limit, 
            self.retries, 
            self.csv_dir)
        
        self.next(self.clean_data)

    @step
    def clean_data(self):
        print("Cleaning data...")
        self.clean_dataframe = clean_city_data(self.dataframe)
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Training model...")
        self.model = train_RandomForestRegressor_model(self.clean_dataframe)
        self.next(self.predict_and_plot)

    @step
    def predict_and_plot(self):
        print("Running Predict and Plot...")
        
        city_predict_and_plot(
            self.model, 
            self.clean_dataframe, 
            self.city, 
            self.img_dir, 
            self.num_prediction_days)
        
        self.next(self.end)

    @step
    def end(self):
        print("AQI prediction pipeline complete.")
        
if __name__ == "__main__":
    AQIPredictionFlow()