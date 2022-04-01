# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from hashlib import sha256
from collections import defaultdict
from datetime import datetime, timedelta

hypers = {
        'vocab_size': 2 ** 14,
        'sample_size': 2 ** 4,
        'batch_size': 2 ** 4
        }

paths = {
        'tokenizer': 'bert-base-multilingual-cased',
        'toks': '../data/wiki/toks',
        'text': '../data/wiki/text',
        'days': '../data/wiki/days',
        'ess': '../data/ess/ESS1-9e01_1.csv', # redownload
        'factors': '../data/ess/factors.json'
        }

langs = "da,de,et,fi,fr,is,it,nl,no,pl,sv"
lang_to_country = {lang: lang.upper() for lang in langs.split(',')}
lang_to_country['es'] = 'SP'
lang_to_country['et'] = 'ES'
lang_to_country['sv'] = 'SE'
lang_to_country['da'] = 'DK'

# get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")

# month 2 ess
def month_to_ess(lang, date, ess):
    f = '%Y/%m/%d'
    rounds_to_date = {"7": "2014/12/31", "8": "2016/12/31", "9": "2018/12/31"} # round release assummption
    best_ess_round = (None, 10000)
    rounds = list(ess[lang_to_country[lang]].keys())
    for r, time in rounds_to_date.items():
        if r not in rounds:
            continue
        if time == date:
            best_ess_round = (r, 0)
            continue
        delta = int(str(datetime.strptime(date, f) - datetime.strptime(time, f)).split()[0].replace('-', ''))
        if delta < best_ess_round[1]:
            best_ess_round = (r, delta)
    vec = ess[lang_to_country[lang]][best_ess_round[0]]
    Y = torch.tensor([vec['avg'], vec['var']]).T
    return Y
           

# connect to digital ocean spaces
def get_s3():
    session = Session()
    client = session.client('s3',
            region_name='AMS3',
            endpoint_url='https://virian.ams3.digitaloceanspaces.com',
            aws_access_key_id=os.getenv("DIGITAL_OCEAN_SPACES_KEY"),
            aws_secret_access_key=os.getenv("DIGITAL_OCEAN_SPACES_SECRET"))
    return client


# load ess, wiki text or wiki days
def load(target):
    files = [f for f in os.listdir(paths[target]) if f[-4:] == 'json']
    data = {}
    for file in files:
        with open(f"{paths[target]}/{file}", 'r') as f:
            if target == 'text':
                data[file[:2]] = defaultdict(lambda: {"text":""}, json.load(f))
            if target == 'toks':
                data[file[:2]] = defaultdict(lambda: [0 for _ in range(hypers['sample_size'])], json.load(f))
            if target == 'days':
                days = [json.loads(line) for line in f]
                for  day in days:
                    data[f"{file[:2]}-{day['date']}"] = day
    return data


# tokenze
def tokenize(batch, tokenizer):
    X = tokenizer(batch, truncation=True, padding="max_length", return_tensors='pt', max_length=hypers['sample_size'])
    X = X['input_ids']
    return X.float()


get_tokenizer = lambda: AutoTokenizer.from_pretrained(paths["tokenizer"])
title_hash = lambda title: sha256((title).encode('utf-8')).hexdigest()
def get_ess():
    with open(paths['factors'], 'r') as f:
        return json.load(f)

