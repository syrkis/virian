# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Tokenizer, Model, train, get_articles, get_dailies, make_months
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

def run_months():
    make_months('da')

def run_training():
    tokenizer = Tokenizer(trained=False)
    exit()
    ds = Dataset()
    for s in ds:
        print(s)
        break
    exit()
    tokenizer = Tokenizer(trained=True)
    loader = DataLoader(dataset=ds, batch_size=30)
    topic_model = TopicModel(ds.vocab_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(topic_model.parameters())
    train(loader, topic_model, optimizer, criterion)

def get_args():
    parser = argparse.ArgumentParser(description="Virian data scraper scripts")
    parser.add_argument('--dailies',  action='store_true', help='scrape daily top read')
    parser.add_argument('--months',  action='store_true', help='make monthly wiki ess data')
    parser.add_argument('--articles', action='store_true', help="scrape wiki articles")
    parser.add_argument('--train', action='store_true', help="scrape wiki articles")
    return parser.parse_args()

# call stack
def main():
    args = get_args()
    langs = 'da no sv nl pl fi it de'.split()
    if args.dailies:
        run_dailies(langs)    
    if args.articles:
        run_articles(langs)
    if args.months:
        run_months()
    if args.train:
        run_training()

if __name__ == "__main__":
    main()

