# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
import argparse
import os
import mlflow


# global variables
variables       = { 'pad': 10 ** 6, 'date_format': "%Y_%m_%d", 'data_dir': '../data', 'data_dirs': ['wiki', 'ess'] }

lang_splits     = { 'train': ['cs','et','fi','fr','de','hu','lt','nl','no','pl','pt','si','es','sv'],
                    'test': ['bg','hr','da','is','he','it','lv','ru','sk'] }

lang_to_country = { 'bg':'bg','hr':'hr','cs':'cz','da':'dk','et':'ee','fi':'fi','fr':'fr','de':'de',
                    'hu':'hu','is':'is','he':'il','it':'it','lv':'lv','lt':'lt','nl':'nl','no':'no',
                    'pl':'pl','pt':'pt','ru':'ru','sk':'sk','si':'si','es':'es','sv':'se' }

ess_cols        = { 'meta' : ['essround','cntry'], 'questions': ["health", "hlthhmp", "rlgblg", "rlgdnm", "rlgblge",
                    "rlgdnme", "rlgdgr", "rlgatnd", "pray", "happy", "sclmeet", "inprdsc", "sclact", "crmvct",
                    "aesfdrk", "ipcrtiv", "imprich", "ipeqopt", "ipshabt", "impsafe", "impdiff", "ipfrule",
                    "ipudrst", "ipmodst", "ipgdtim", "impfree", "iphlppl", "ipsuces", "ipstrgv", "ipadvnt",
                    "ipbhprp", "iprspot", "iplylfr", "impenv", "imptrad", "impfun"] }

# get args
def get_args():
    parser = argparse.ArgumentParser(description="Virian Script")
    parser.add_argument('--wiki',  action='store_true')
    parser.add_argument('--ess', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dataset', action='store_true')
    parser.add_argument('--local', action='store_true')

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vocab-size", type=int, default=10 ** 6)
    parser.add_argument("--embedding-dim", type=int, default=300)
    return parser.parse_args()


# get traning parameters
def get_params(args):
    params = { "Batch Size": args.batch_size,
               "Sample Size": args.sample_size,
               "Vocab Size" : args.vocab_size,
               "Embedding Dim" : args.embedding_dim,
               "Learning Rate": args.lr }
    return params


# connect to digital ocean spaces
def get_s3():
    session = Session()
    client = session.client('s3', region_name='AMS3',
                            endpoint_url='https://virian.ams3.digitaloceanspaces.com',
                            aws_access_key_id=os.getenv("DIGITAL_OCEAN_SPACES_KEY"),
                            aws_secret_access_key=os.getenv("DIGITAL_OCEAN_SPACES_SECRET"))
    return client


# cycle through iterator (for validation)
def cycle(seq):
    while True:
        for X, W, Y in seq:
            yield X, W, Y


