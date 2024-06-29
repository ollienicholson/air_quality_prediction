from metaflow import FlowSpec, step

from main import fetch_data, clean_data, train_model, predict_and_plot


class AQIPredictionFlow(FlowSpec):

    @step
    def start(self):
        self.city = 'Sydney'
        self.start_date = '2021-01-01'
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