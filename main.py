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


# runners
def run_dailies(langs):
    pool = Pool(processes=len(langs))
    pool.map(get_dailies, langs)

def run_articles(langs):
    pool = Pool(processes=len(langs))
    pool.map(get_articles, langs)

def run_months(langs):
    pool = Pool(processes=len(langs))
    pool.map(make_months, langs)

def run_training():
    tokenizer = Tokenizer(trained=True)
    ds = Dataset(tokenizer)
    model = Model(ds.sample_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    train(ds, model, optimizer, criterion)

def get_args():
    parser = argparse.ArgumentParser(description="Virian data scraper scripts")
    parser.add_argument('--dailies',  action='store_true', help='scrape daily top read')
    parser.add_argument('--months',  action='store_true', help='make monthly wiki ess data')
    parser.add_argument('--articles', action='store_true', help="scrape wiki articles")
    parser.add_argument('--values', action='store_true', help="focus on ess data")
    parser.add_argument('--train', action='store_true', help="scrape wiki articles")
    parser.add_argument('--langs', default="de fi da no sv nl pl it", help="what langs to taget")
    return parser.parse_args()

# call stack
def main():
    args = get_args()
    langs = args.langs.split(",") # de fi not done
    if args.dailies:
        run_dailies(langs)    
    if args.articles:
        run_articles(langs)
    if args.months:
        run_months(langs)
    if args.train:
        run_training()
    if args.values:
        load_df()

if __name__ == "__main__":
    main()

