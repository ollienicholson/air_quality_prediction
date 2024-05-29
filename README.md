
# Air Quality Index (AQI) Prediction Project

This project aims to analyze historical air quality data, identify trends, and build a model to predict future AQI levels. The project leverages Metaflow for workflow orchestration, alongside other Python libraries for data manipulation, visualization, and machine learning.

## Project Overview

The main steps of this project include:

1. **Data Ingestion**: Collecting air quality data from the OpenAQ API.
2. **Data Cleaning and Preprocessing**: Cleaning and preprocessing the data for analysis.
3. **Data Analysis**: Performing exploratory data analysis (EDA) to identify trends and patterns.
4. **Model Training**: Training a machine learning model to predict AQI.
5. **Model Evaluation**: Evaluating the model's performance.
6. **Prediction and Visualization**: Making predictions and visualizing the results.

## Requirements

To run this project, you need the following Python packages:

- requests
- pandas
- scikit-learn
- matplotlib
- seaborn
- metaflow

You can install these packages using the provided `requirements.txt` file:

```sh
pip freeze >> requirements.txt

pip install -r requirements.txt
```

## Project Structure

```
.
├── main.py                 # Metaflow pipeline script
├── requirements.txt        # List of required packages
├── README.md               # Project README file
└── data                    # Directory to store fetched data (optional)
```

## Data Ingestion

Data is collected from the OpenAQ API using the `fetch_data` function, which retrieves air quality measurements for a specified city and date range.

## Data Cleaning and Preprocessing

The `clean_data` function processes the raw data by:
- Extracting and converting the UTC date.
- Selecting relevant columns.
- Pivoting the table to get parameters as columns.
- Resampling to daily averages.
- Dropping rows with missing values.

## Model Training

The `train_model` function trains a RandomForestRegressor on the cleaned data to predict PM2.5 levels, using the day of the year as the feature.

## Prediction and Visualization

The `predict_and_plot` function generates future dates, predicts their AQI values using the trained model, and plots both historical and predicted AQI values.

## Metaflow Pipeline

The entire workflow is orchestrated using Metaflow, structured in the `AQIPredictionFlow` class:

```python
from metaflow import FlowSpec, step

class AQIPredictionFlow(FlowSpec):

    @step
    def start(self):
        self.city = 'Los Angeles'
        self.start_date = '2023-01-01'
        self.end_date = '2023-12-31'
        self.next(self.fetch_data)

    @step
    def fetch_data(self):
        self.data = fetch_data(self.city, self.start_date, self.end_date)
        self.next(self.clean_data)

    @step
    def clean_data(self):
        self.cleaned_data = clean_data(self.data)
        self.next(self.train_model)

    @step
    def train_model(self):
        self.model = train_model(self.cleaned_data)
        self.next(self.predict_and_plot)

    @step
    def predict_and_plot(self):
        predict_and_plot(self.model, self.cleaned_data)
        self.next(self.end)

    @step
    def end(self):
        print("AQI prediction pipeline complete.")

if __name__ == '__main__':
    AQIPredictionFlow()
```

## How to Run

1. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the Metaflow pipeline**:
    ```sh
    python main.py run
    ```

## Results

The project outputs include:

- Cleaned and preprocessed air quality data.
- Visualizations of historical AQI trends.
- Predictions of future AQI levels and their visualization.

## Acknowledgements

This project uses data from the OpenAQ API.

---
