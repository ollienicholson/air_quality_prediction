from metaflow import FlowSpec, step

from main import fetch_data, clean_data, train_model, predict_and_plot


class AQIPredictionFlow(FlowSpec):
    '''
    NOTE: need to update class description
    '''

    @step
    def start(self):
        print("Starting data flow...")
        self.city = 'Melbourne'
        self.start_date = '2021-01-01'
        self.end_date = '2023-12-31'
        self.next(self.fetch_data)

    @step
    def fetch_data(self):
        print("Fetching data...")
        self.data = fetch_data(self.city, self.start_date, self.end_date)
        self.next(self.clean_data)

    @step
    def clean_data(self):
        print("Cleaning data...")
        self.cleaned_data = clean_data(self.data)
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Training model...")
        self.model = train_model(self.cleaned_data)
        self.next(self.predict_and_plot)

    @step
    def predict_and_plot(self):
        print("Running Predict and Plot...")
        predict_and_plot(self.model, self.cleaned_data)
        self.next(self.end)

    @step
    def end(self):
        print("AQI prediction pipeline complete.")