# month.py
#   class for a virian month
# by: Noah Syrkis

# imports
from datetime import datetime, timedelta


# script class
class Month:
    def __init__(self, api, month):
        self.api = api

    def analyze(self):

        print(self.api)
