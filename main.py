# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
import os
from bpemb import BPEmb


# get args
def get_args():
    parser = argparse.ArgumentParser(description="Virian Script")
    parser.add_argument('--wiki',  action='store_true', help='sustain wiki')
    parser.add_argument('--ess', action='store_true', help="sustain ess")
    parser.add_argument('--train', action='store_true', help="train model")
    parser.add_argument('--dataset', action='store_true', help="explore dataset")
    parser.add_argument('--tokenize', action='store_true', help="token files")
    parser.add_argument('--readme', action="store_true", help="readme")
    parser.add_argument('--local', action="store_true", help="local run?")
    parser.add_argument('--langs', help="target specific languages (wiki)")
    return parser.parse_args()


def run_tokenize(langs):
    tokenizer = BPEmb(lang="multi", vs=10 ** 6, dim=300)
    texts = load('text', langs) 
    for lang, articles in texts.items():
        toks = {}
        for title, text in tqdm(articles.items()):
            if 'text' in text:
                toks[title] = tokenizer.encode_ids(text['text'])
        with open(f"{paths['toks']}/{lang}.json", 'w') as f:
            json.dump(toks, f)

def run_wiki(langs):
    with Pool(len(langs)) as p:
        p.map(get_dailies, langs)
    with Pool(len(langs)) as p:
        p.map(get_articles, langs)

def run_ess():
    ess = ESS()
    out = ess.get_target('SE', '2020_10_30')
    print(out)

def run_train(langs, local):
    model     = Model()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds        = Dataset(langs[0], local)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    writer    = SummaryWriter()
    model     = train(ds, model, optimizer, criterion, device, writer)
    writer.flush()


# call stack
def main():
    # setup()
    args = get_args()
    if args.ess:
        run_ess()    
    if args.wiki:
        train_langs, test_langs = get_langs()
        run_wiki(train_langs + test_langs)
    if args.train:
        train_langs, test_langs = get_langs()
        run_train((train_langs, test_langs), args.local)
    if args.dataset:
        train_langs, _ = get_langs()
        if args.local:
            ds = Dataset(train_langs[:1])
        for X, W, Y in ds:
            print(X, W, Y)
    if args.tokenize:
        train_langs, test_langs = get_langs()
        run_tokenize(train_langs + test_langs)

if __name__ == "__main__":
    main()

