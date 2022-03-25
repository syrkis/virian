# months.py
#   constructs virian monthly datasets from wiki and ess
# by: Noah Syrkis

# imports
import os, json
import torch
from tqdm import tqdm
from collections import defaultdict
from hashlib import sha256
import pandas as pd
import numpy as np


# make all months for specific lang
def make_months(lang_and_values):
    lang, values = lang_and_values
    dailies_dir = f"../data/dailies/{lang}"
    months = sorted(list(set([f[:7] for f in os.listdir(dailies_dir)])))
    for month in tqdm(months):
        month_value = month_value_mapper(values, month)
        if month_value:
            data = make_month(lang, month_value, month)
            with open(f"../data/months/{lang}_{month}.json", 'w') as f:
                json.dump(data, f)


# create month
def make_month(lang, month_value, month): # TODO: infer monthly ess factor dists
    make_month_new(lang, month_value, month)
    dailies_dir = f"../data/dailies/{lang}"
    dailies = [f for f in os.listdir(dailies_dir) if f[:7] == month]
    dailies_data = {daily[:10]: defaultdict(lambda: 0) for daily in dailies}
    for daily in dailies:
        with open(f"{dailies_dir}/{daily}", 'r') as f:
            day = daily[:10]
            articles = json.loads(f.read())
            for article in articles:
                title_hash = sha256((article['article']).encode('utf-8')).hexdigest()
                dailies_data[day][title_hash] += article['views']
    data = {
            "month": month,
            "values": {"mean": list(month_value[0]), "var": list(month_value[1])},
            "lang": lang, "dailies": {k: None for k in dailies_data.keys()}
            }
    for k1, v1 in dailies_data.items():
        data["dailies"][k1] = {k2: v2 for k2, v2 in v1.items()}
    return data


# constuct compelte month sample
def make_month_new(lang, values, month):
    X = torch.zeros((31, 1000))
    W = torch.zeros((31, 1000))
    Y = torch.zeros((2, 5))
    dailies_dir = f"../data/dailies/{lang}"
    dailies = [f for f in os.listdir(dailies_dir) if f[:7] == month]
    dailies_data = {daily[:10]: defaultdict(lambda: 0) for daily in dailies} # daily articles W
    for daily in dailies:
        with open(f'{dailies_dir}/{daily}', 'r') as f:
            pass       


# function that assigns values to a month.....
def month_value_mapper(values, month):
    out = None
    if month < "2015_12" and "7" in values: # if ess exist and month date given
        out = values['7']   
    elif month < "2017_06" and "8" in values:
        out = values['8']   
    elif month < "2019_06" and "9" in values:
        out = values['9']
    return out
    
