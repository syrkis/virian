# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import json
from torch.utils.tensorboard import SummaryWriter


# get args
def get_args():
    parser = argparse.ArgumentParser(description="Virian Script")
    parser.add_argument('--wiki',  action='store_true', help='sustain wiki')
    parser.add_argument('--ess', action='store_true', help="sustain ess")
    parser.add_argument('--train', action='store_true', help="train model")
    parser.add_argument('--dataset', action='store_true', help="explore dataset")
    parser.add_argument('--tokenize', action='store_true', help="token files")
    parser.add_argument('--readme', action="store_true", help="readme")
    parser.add_argument('--langs', help="target specific languages (wiki)")
    return parser.parse_args()


# runners
def run_tokenize():
    texts = load('text') 
    tokenizer = get_tokenizer()
    for lang, articles in texts.items():
        toks = {}
        for title, text in tqdm(articles.items()):
            toks[title] = tokenize(text['text'], tokenizer)[0].tolist()
        with open(f"{paths['toks']}/{lang}.json", 'w+') as f:
            json.dump(toks, f)

def run_wiki(langs):
    with Pool(2) as p:
        p.map(get_dailies, langs)
    with Pool(2) as p:
        p.map(get_articles, langs)

def run_ess():
    construct_factors()    

def run_dataset():
    ds = Dataset()
    loader = DataLoader(dataset=ds, batch_size=64)
    for X, W, Y in tqdm(loader):
        print(Y)
        break

def run_train():
    ds = Dataset()
    loader = DataLoader(dataset=ds, batch_size=hypers['batch_size'])
    model = Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter()
    model = train(loader, model, optimizer, criterion, writer)
    writer.flush()


# call stack
def main():
    args = get_args()
    langs = parse_readme_langs()
    if args.ess:
        run_ess()    
    if args.wiki:
        run_wiki(langs)
    if args.train:
        run_train()
    if args.dataset:
        run_dataset()
    if args.tokenize:
        run_tokenize()

if __name__ == "__main__":
    main()

