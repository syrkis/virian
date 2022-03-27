# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
import os
import json
from tqdm import tqdm


# connect to digital ocean spaces
def get_s3():
    session = Session()
    client = session.client('s3',
            region_name='AMS3',
            endpoint_url='https://virian.ams3.digitaloceanspaces.com',
            aws_access_key_id=os.getenv("DIGITAL_OCEAN_SPACES_KEY"),
            aws_secret_access_key=os.getenv("DIGITAL_OCEAN_SPACES_SECRET"))
    return client


def load(target):
    path  = f"../data/wiki/{target}" 
    files = [f for f in os.listdir(path) if f[-4:] == 'json']
    data = {}
    for file in files:
        with open(f"{path}/{file}", 'r') as f:
            if target == 'text':
                data[file[:2]] = json.load(f)
            if target == 'days':
                data = [json.loads(line) for line in f]
    return data

