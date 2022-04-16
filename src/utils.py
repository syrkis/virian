# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
import argparse
import os


# global variables
variables       = { 'pad': 10 ** 6, 'date_format': "%Y_%m_%d", 'data_dir': '../data', 'data_dirs': ['wiki', 'ess'] }
parameters      = { 'vocab_size': 10 ** 6, 'sample_size': 2 ** 4, 'embedding_dim': 300, 'batch_size': 2 ** 3 }
lang_splits     = { 'train': ['cs','et','fi','fr','de','hu','lt','nl','no','pl','pt','si','es','sv'], 'test': ['bg','hr','da','is','he','it','lv','ru','sk'] }
lang_to_country = { 'bg':'bg','hr':'hr','cs':'cz','da':'dk','et':'ee','fi':'fi','fr':'fr','de':'de','hu':'hu','is':'is','he':'il','it':'it',
                    'lv':'lv','lt':'lt','nl':'nl','no':'no','pl':'pl','pt':'pt','ru':'ru','sk':'sk','si':'si','es':'es','sv':'se' }

# get args
def get_args():
    parser = argparse.ArgumentParser(description="Virian Script")
    parser.add_argument('--wiki',  action='store_true')
    parser.add_argument('--ess', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dataset', action='store_true')
    parser.add_argument('--local', action="store_true")
    return parser.parse_args()

# connect to digital ocean spaces
def get_s3():
    session = Session()
    client = session.client('s3', region_name='AMS3',
                            endpoint_url='https://virian.ams3.digitaloceanspaces.com',
                            aws_access_key_id=os.getenv("DIGITAL_OCEAN_SPACES_KEY"),
                            aws_secret_access_key=os.getenv("DIGITAL_OCEAN_SPACES_SECRET"))
    return client
