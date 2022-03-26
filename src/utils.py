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

def big_daily(lang):
    dailies_dir = f'../data/dailies/{lang}'
    files = sorted(os.listdir(dailies_dir))
    D = {}
    for file in tqdm(files):
        with open(f"{dailies_dir}/{file}", 'r') as f:
            data = json.load(f)
            D[file[:-5]] = data
    with open(f'../data/dailies_new/{lang}.json', 'w') as f:
        json.dump(D, f, indent=2)



