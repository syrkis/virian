# utils.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session
import argparse
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate


# global variables
variables       = { 'pad': 10 ** 6, 'date_format': "%Y_%m_%d", 'data_dir': 'data', 'data_dirs': ['wiki', 'ess'] }

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
    parser.add_argument('--model', action='store_true')
    parser.add_argument('--dataset', action='store_true')
    parser.add_argument('--local', action='store_true')
    # model params
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vocab-size", type=int, default=10 ** 6)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=300)
    return parser.parse_args()


# get traning parameters
def get_params(args):
    langs  = lang_splits['train'][:3] if args.local else lang_splits['train'] + lang_splits['test']
    params = { "Batch Size": args.batch_size,
               "Sample Size": args.sample_size,
               "Vocab Size" : args.vocab_size,
               "Embedding Dim" : args.embedding_dim,
               "Latent Dim" : args.latent_dim,
               "Learning Rate": args.lr,
               "Languages": langs}
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


# get metrics for labml (accuracy and MSE)
def get_metrics(x_loss, y_loss, x_loss_val, y_loss_val, train_acc, valid_acc, params):

    metrics = {'wiki train mse': x_loss.item() / params["Batch Size"],
               'wiki valid mse': x_loss_val.item() / params["Batch Size"],

                'ess train mse': y_loss.item() / params["Batch Size"],
                'ess valid mse': y_loss_val.item() / params["Batch Size"],

                'ess train acc': train_acc,
                'ess valid acc': valid_acc}
    return metrics

