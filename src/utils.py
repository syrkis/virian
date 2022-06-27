# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
#from boto3.session import Session
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

# global variables
variables       = {'pad':10 **6, 'date_format': "%Y_%m_%d", 'data_dir': 'data', 'data_dirs': ['wiki', 'ess']}

# get args
def get_args():
    parser = argparse.ArgumentParser(description="Virian Script")
    parser.add_argument('--wiki',  action='store_true')
    parser.add_argument('--ess', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', action='store_true')
    parser.add_argument('--dataset', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--config', action='store_true')
    # model params
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--target", type=str, default="human_values")
    return parser.parse_args()


# get traning parameters
def get_params(args):
    # langs  = local_langs if args.local else lang_splits['train']
    params = { "Batch Size": args.batch_size, # rename to config
               # "Sample Size": args.sample_size,
               "Embedding Dim" : args.embedding_dim,
               "Learning Rate": args.lr,
               "Target" : args.target}
               # "Languages": langs}
    return params


def get_conf():
    with open('config.json', 'r') as f:
        return json.load(f)


# connect to digital ocean spaces
"""
def get_s3():
    session = Session()
    client = session.client('s3', region_name='AMS3',
        endpoint_url='https://virian.ams3.digitaloceanspaces.com',
        aws_access_key_id=os.getenv("DIGITAL_OCEAN_SPACES_KEY"),
        aws_secret_access_key=os.getenv("DIGITAL_OCEAN_SPACES_SECRET"))
    return client
"""


# cycle through iterator (for validation)
def cycle(seq):
    while True:
        for X, W, Y in seq:
            yield X, W, Y


# make k fold loaders
def cross_validate(ds, lang, params, device):
    train_idx, valid_idx  = ds.k_fold(lang)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader  = get_loader(ds, params['Batch Size'], train_sampler, device)
    valid_loader  = get_loader(ds, params['Batch Size'], valid_sampler, device)
    valid_iter    = cycle(valid_loader)
    return train_loader, valid_iter


# get loader for train and valid (and test) data sets
def get_loader(ds, batch_size, sampler, device):
    loader = DataLoader(dataset=ds, batch_size=batch_size, sampler=sampler, drop_last=True,
        collate_fn=lambda x: list(map(lambda x: x.to(device), default_collate(x))))
    return loader


# put tensors to device
def to_device(x, w, y, device):
    x = default_collate(x).to(device)
    w = default_collate(w).to(device)
    y = default_collate(y).to(device)
    return x, w, y

# how often does model beat baseline?
def baseline(y_val_pred, Y_val):
    pred_dist = torch.abs(Y_val - y_val_pred)
    base_dist = torch.abs(Y_val)
    base      = torch.sum(pred_dist <= base_dist) / torch.numel(Y_val)
    return base






