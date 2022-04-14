# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from hashlib import sha256
from collections import defaultdict
from datetime import datetime, timedelta



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

def get_embeddings():
    return AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased").bert.embeddings.word_embeddings

hypers = {
        'vocab_size': 2 ** 14,
        'sample_size': 2 ** 4,
        'embedding_dim': 768,
        'batch_size': 2 ** 3
        }

paths = {
        'tokenizer': 'bert-base-multilingual-cased',
        'toks': '../data/wiki/toks',
        'text': '../data/wiki/text',
        'days': '../data/wiki/days',
        'ess': '../data/ess/ESS1-9e01_1.csv', # redownload
        'factors': '../data/ess/factors.json',
        "all_dirs": ['../data', '../data/models', '../data/ess', '../data/wiki', '../data/wiki/days', '../data/wiki/toks', '../data/wiki/text']
        }

# get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")

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
def load(target, langs, local=False):
    files = [f for f in os.listdir(paths[target]) if f[-4:] == 'json']
    data = {}
    for idx, file in enumerate(files):
        if file[:2] in langs: # train or test split
            with open(f"{paths[target]}/{file}", 'r') as f:
                if target == 'text':
                    data[file[:2]] = defaultdict(lambda: {"text":""}, json.load(f))
                if target == 'toks':
                    data[file[:2]] = defaultdict(lambda: [0 for _ in range(hypers['sample_size'])], json.load(f))
                if target == 'days':
                    days = [json.loads(line) for line in f]
                    for  day in days:
                        data[f"{file[:2]}-{day['date']}"] = day
        if local and idx >= 2:
            break
    return data


# tokenze
def tokenize(batch, tokenizer):
    X = tokenizer(batch, truncation=True, padding="max_length", return_tensors='pt', max_length=hypers['sample_size'])
    X = X['input_ids']
    return X


def get_tokenizer():
    return AutoTokenizer.from_pretrained(paths["tokenizer"])

title_hash = lambda title: sha256((title).encode('utf-8')).hexdigest()

def get_ess():
    with open(paths['factors'], 'r') as f:
        return json.load(f)


def get_langs():
    with open('README.md', 'r') as f:
        table = f.read().split("## Countries")[1].strip().lower()
    data        = [line.strip().split('|')[1:-1] for line in table.split('\n')[2:]]
    data        = [[entry.strip() for entry in line] for line in data]
    train_langs = [line[2] for line in data if "".join(line[-3:]) == "xxx"]
    test_langs  = [line[2] for line in data if line[2] not in train_langs]
    return train_langs, test_langs


def get_lang2cntry():
    with open('README.md', 'r') as f:
        table = f.read().split("## Countries")[1].strip().lower()
    out = table.split('\n')[2:]
    out = [[e.strip() for e in line.split('|')[2:4]] for line in out]
    D = {}
    for country, lang in out:
        D[lang] = country
    return D

# lang_to_cntry = get_lang2cntry()
