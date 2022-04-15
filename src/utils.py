# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
import torch
import os
import json
from tqdm import tqdm
from hashlib import sha256
from collections import defaultdict
from datetime import datetime, timedelta


hypers = {
        'vocab_size': 119547, # from bert tokenizer
        'sample_size': 2 ** 4,
        'embedding_dim': 10,
        'batch_size': 2 ** 3
        }







title_hash = lambda title: sha256((title).encode('utf-8')).hexdigest()

def get_langs():
    with open('README.md', 'r') as f:
        table = f.read().split("## Countries")[1].strip().lower()
    data        = [line.strip().split('|')[1:-1] for line in table.split('\n')[2:]]
    data        = [[entry.strip() for entry in line] for line in data]
    train_langs = [line[2] for line in data if "".join(line[-3:]) == "xxx"]
    test_langs  = [line[2] for line in data if line[2] not in train_langs]
    return train_langs, test_langs

# lang_to_cntry = get_lang2cntry()
paths = {
        'tokenizer': 'bert-base-multilingual-cased',
        'toks': '../data/wiki/toks',
        'text': '../data/wiki/text',
        'days': '../data/wiki/days',
        'ess': '../data/ess/ESS1-9e01_1.csv', # redownload
        'factors': '../data/ess/factors.json',
        "all_dirs": [
            '../data', '../data/models', '../data/ess',
            '../data/wiki', '../data/wiki/days', '../data/wiki/toks',
            '../data/wiki/text'
            ]
        }


# runners
def setup():
    for path in paths['all_dirs']:
        if not os.path.exists(path):
            os.makedirs(path)
    for lang in parse_readme_langs():
        for idx, path in [hypers['days'], hypers['text']]:
            file = f"{path}/{lang}.json"
            if not os.path.exists(file):
                if idx == 0:
                    fp = open(file, 'x'); fp.close()
                else:
                    with open(file, 'w') as f:
                        json.dump({"__failed__": []}, f) # sould have been set and in own level

# connect to digital ocean spaces
def get_s3():
    session = Session()
    client = session.client('s3',
            region_name='AMS3',
            endpoint_url='https://virian.ams3.digitaloceanspaces.com',
            aws_access_key_id=os.getenv("DIGITAL_OCEAN_SPACES_KEY"),
            aws_secret_access_key=os.getenv("DIGITAL_OCEAN_SPACES_SECRET"))
    return client

# get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")
