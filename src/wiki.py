# wiki.py
#   class script for given given language  
# by: Noah Syrkis


# script class
class Wiki:

    api = lambda wiki, year, month, day: f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{wiki}/all-access/{year}/{month}/{day}"

    def __init__(self, wiki):
        self.wiki = wiki
