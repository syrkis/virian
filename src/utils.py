# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
import os
import json
from tqdm import tqdm
from hashlib import sha256

hypers = {'vocab_size': 2 ** 14, 'sample_size': 2 ** 7}
paths = {
        'tokenizer': '../data/model/tokenizer.json',
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
                data[file[:2]] = json.load(f)
            if target == 'days':
                data = [json.loads(line) for line in f]
    return data


# iterate through raw text (for tokenizer)
def text_iter():
    files = [f for f in os.listdir(path) if f[-4:] == 'json']
    for file in files:
        with open(f"{paths['text']}/{file}", 'r') as f:
            data = json.load(f)
            for sample in data.values():
                yield sample['text']


# hash title name of wiki text
def title_hash(title):
    return sha256((article['article']).encode('utf-8')).hexdigest()
