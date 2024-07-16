

class cityClass:
    
    def __init__(self, city: str, 
                 start_date: str, 
                 end_date: str, 
                 limit: int, 
                 retries: int, 
                 num_prediction_days:int):
        
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        self.limit = limit
        self.retries = retries
        self.num_prediction_days = num_prediction_days
    
    def city_parameters(self):
        return self.city, self.start_date, self.end_date, self.limit, \
            self.retries, self.num_prediction_days
            

# Assign variables
city = 'Melbourne'
start_date = '2021-01-01'
end_date = '2023-12-31'
limit = 12000
retries = 3
num_prediction_days = 360 # in days


city_var = cityClass("Melbourne", "2021-01-01", "2023-12-31", "12000", "3", "360")

print(city_var.start_date)
