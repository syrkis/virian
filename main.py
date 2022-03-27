# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import *
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool
import argparse


# get cliargs
def get_args():
    parser = argparse.ArgumentParser(description="Virian Script")
    parser.add_argument('--wiki',  action='store_true', help='create/update wiki data')
    parser.add_argument('--ess', action='store_true', help="run ess script")
    parser.add_argument('--train', action='store_true', help="scrape wiki articles")
    parser.add_argument('--dataset', action='store_true', help="explore dataset")
    parser.add_argument('--langs', default="de,fi,da,no,sv,nl,pl,it,et,fr,is", help="target langs")
    return parser.parse_args()


# runners
def run_wiki(langs):
    with Pool(2) as p:
        p.map(get_dailies, langs)
    with Pool(2) as p:
        p.map(get_articles, langs)

def run_ess():
    construct_factors()    

def run_dataset(langs):
    tokenizer = Tokenizer(trained=True)
    ds = Dataset(tokenizer)
    for sample in ds:
        print(sample.shape, end=' ')
    
def run_train():
    tokenizer = Tokenizer(trained=False)
    ds = Dataset(tokenizer)
    model = Model(ds.sample_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    train(ds, model, optimizer, criterion)


# call stack
def main():
    args = get_args()
    langs = args.langs.split(",")
    if args.ess:
        run_ess()    
    if args.wiki:
        run_wiki(langs)
    if args.train:
        run_train()
    if args.dataset:
        run_dataset(langs)

if __name__ == "__main__":
    main()

