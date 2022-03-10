# dailies.py
#   scarpes wiki project daily top 1000 api
# by: Noah Syrkis

# imports
from datetime import datetime, timedelta
import os, json, requests, time
from tqdm import tqdm


# run time function
def get_dailies(lang):
    dailies_dir = f"../data/dailies/{lang}"
    api = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
    files = os.listdir(dailies_dir)
    done = sorted([file[:-5].replace('_', '/') for file in files if file[-5:] == ".json"])
    start_date = get_date(done[-1] if done else "2015/07/01")
    end_date = datetime.now() - timedelta(days=3) 
    date = start_date
    for _ in tqdm(range((end_date - start_date).days)):
        date_str = get_str(date)
        headers = {"User-Agent": "nobr@itu.dk"}
        url = get_url(api, date_str, lang)
        res = requests.get(url, headers=headers).text
        data = json.loads(res)['items'][0]['articles']
        with open(f"{dailies_dir}/{get_str(date).replace('/', '_')}.json", 'w') as f:
            json.dump(data, f)
        date += timedelta(days = 1)


# helpers
get_url = lambda api, date, lang: f'{api}/{lang}.wikipedia.org/all-access/{date}'
get_date = lambda s: datetime.strptime(s, "%Y/%m/%d")
get_str = lambda d: str(d).split()[0].replace('-', '/')


# dev calls
def main():
    pass

if __name__ == "__main__":
    main()
    
