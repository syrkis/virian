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
from transformers import logging
logging.set_verbosity_error()


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
def setup():
    for path in paths['all_dirs']:
        if not os.path.exists(path):
            os.makedirs(path)
    for lang in parse_readme_langs():
        for idx, path in [hypers['days'], hypers['text']]:
            file = f"{path}/{lang}.json"
            if not os.path.exists(file):
                if idx == 0:
                    fp = open(file, 'x'); fp.close()
                else:
                    with open(file, 'w') as f:
                        json.dump({"__failed__": []}, f) # sould have been set and in own level


def run_tokenize(langs):
    texts = load('text', langs) 
    tokenizer = get_tokenizer()
    for lang, articles in texts.items():
        toks = {}
        for title, text in tqdm(articles.items()):
            if 'text' in text:
                toks[title] = tokenize(text['text'], tokenizer)[0].tolist()
        with open(f"{paths['toks']}/{lang}.json", 'w') as f:
            json.dump(toks, f)

def run_wiki(langs):
    with Pool(len(langs)) as p:
        p.map(get_dailies, langs)
    with Pool(len(langs)) as p:
        p.map(get_articles, langs)

def run_ess():
    construct_factors()    

def run_train(langs):
    model     = Model()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds        = Dataset(langs[0])
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
        run_train((train_langs, test_langs))
    if args.dataset:
        run_dataset()
    if args.tokenize:
        train_langs, test_langs = get_langs()
        run_tokenize(train_langs + test_langs)

if __name__ == "__main__":
    main()

