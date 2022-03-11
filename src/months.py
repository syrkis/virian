# months.py
#   constructs virian monthly datasets from wiki and ess
# by: Noah Syrkis

# imports
import os, json
from tqdm import tqdm
from collections import defaultdict
from hashlib import sha256
import pandas as pd


# make all months for specific lang
def make_months(lang_and_values):
    lang, values = lang_and_values
    dailies_dir = f"../data/dailies/{lang}"
    months = sorted(list(set([f[:7] for f in os.listdir(dailies_dir)])))
    for month in tqdm(months):
        data = make_month(lang, values, month)
        with open(f"../data/months/{lang}_{month}.json", 'w') as f:
            json.dump(data, f)


# create month
def make_month(lang, values, month): # TODO: add infered monthly ess factor dists
    dailies_dir = f"../data/dailies/{lang}"
    dailies = [f for f in os.listdir(dailies_dir) if f[:7] == month]
    dailies_data = {daily[:10]: defaultdict(lambda: 0) for daily in dailies}
    for daily in dailies:
        with open(f"{dailies_dir}/{daily}", 'r') as f:
            day = daily[:10]
            articles = json.loads(f.read())
            for article in articles:
                title_hash = sha256((article['article']).encode('utf-8')).hexdigest()
                dailies_data[day][title_hash] += article['views'] # TODO: use title hash
    data = {"month": month, "values": None, "lang": lang, "dailies": {k: None for k in dailies_data.keys()}}
    for k1, v1 in dailies_data.items():
        data["dailies"][k1] = {k2: v2 for k2, v2 in v1.items()}
    return data

# function that assigns values to a month.....

