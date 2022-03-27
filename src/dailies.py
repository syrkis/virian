# dailies.py
#   scarpes wiki project daily top 1000 api
# by: Noah Syrkis

# imports
from datetime import datetime, timedelta
import os, json, requests, time
from tqdm import tqdm


# run time function
def get_dailies(lang):
    file = f"../data/dailies_big/{lang}.json"
    api = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
    dones = get_dones(file)
    start_date = get_date(dones[-1].replace('_', '/') if dones else "2015/07/01")
    end_date = datetime.now() - timedelta(days=3) 
    date = start_date
    for _ in tqdm(range((end_date - start_date).days)):
        if get_str(date).replace('/', '_') in dones:
            date += timedelta(days = 1)
            continue
        date_str = get_str(date)
        headers = {"User-Agent": "nobr@itu.dk"}
        url = get_url(api, date_str, lang)
        res = requests.get(url, headers=headers).text
        data = json.loads(res)['items'][0]['articles']
        for sample in data:
            del sample['rank']
        D = {'date': get_str(date).replace('/', '_'), 'data': data}
        with open(file, 'a+') as f:
            json.dump(D, f)
            f.write('\n')
        date += timedelta(days = 1)


# helpers
get_url = lambda api, date, lang: f'{api}/{lang}.wikipedia.org/all-access/{date}'
get_date = lambda s: datetime.strptime(s, "%Y/%m/%d")
get_str = lambda d: str(d).split()[0].replace('-', '/')

def  get_dones(file):
    exist = os.path.exists(file)
    if exist:
        with open(file, 'r') as f:
            dones = [json.loads(sample)['date'] for sample in f.readlines()]
        return sorted(dones)
    return None


# dev calls
def main():
    pass

if __name__ == "__main__":
    main()
    
