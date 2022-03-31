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

hypers = {
        'vocab_size': 2 ** 14,
        'sample_size': 2 ** 4,
        'batch_size': 2 ** 4}
paths = {
        'tokenizer': 'bert-base-multilingual-cased',
        'toks': '../data/wiki/toks',
        'text': '../data/wiki/text',
        'days': '../data/wiki/days',
        'ess': '../data/ess/raw.csv' # redownload
        }

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
                data[file[:2]] = defaultdict(lambda: torch.zeros(hypers['sample_size']), json.load(f))
            if target == 'days':
                data[file[:2]] = [json.loads(line) for line in f]
    return data


# tokenze
def tokenize(batch, tokenizer):
    X = tokenizer(batch, truncation=True, padding="max_length", return_tensors='pt', max_length=hypers['sample_size'])
    X = X['input_ids']
    return X.float()


get_tokenizer = lambda: AutoTokenizer.from_pretrained(paths["tokenizer"])
title_hash = lambda title: sha256((title).encode('utf-8')).hexdigest()
