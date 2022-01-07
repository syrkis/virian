# region.py
#   virian region class
# by: Noah Syrkis

# imports
from src.month import Month

# script class
class Region:

    months = {}
    
    def __init__(self, country):
        self.country = country
        self.api = self._get_api()
    
    def _get_api(self):
        root = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
        if self.country:
            return f"{root}/top-per-country/{self.country}/all-access"
        return f"{root}/top/en.wikipedia.org/all-access"
    
    def analyze(self, _month):
        month = Month(f"{self.api}/{_month}")
        month.analyze()
        
        
