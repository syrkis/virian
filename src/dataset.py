# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from torch.utils.data import Dataset, IterableDataset
from itertools import cycle
from src.utils import get_s3
import pandas as pd
import os, json, random, torch
from tqdm import tqdm
import random


# wiki dataset
class Dataset(IterableDataset):

    vocab_size = 2 ** 14
    sample_size = 2 ** 8
    months_dir = "../data/months"
    dailies_dir = "../data/dailies_new"
    articles_dir = "../data/articles"
    monthly_files = [f for f in os.listdir(months_dir) if f[-5:] == ".json"]
    article_files = [f for f in os.listdir(articles_dir) if f[-5:] == ".json"]
    s3 = get_s3()

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        
    def get_stream(self, files):
        return cycle(self.process_data(files))

    def process_data(self, files):
        if self.tokenizer: # training model # MAYBE LOAD PRE BUILD TENSOR DATASET
            files = os.listdir(self.dailies_dir) 
            for file in files:
                with open(f"{self.dailies_dir}/{file}", 'r') as f:
                    D = json.load(f)
                    yield D.keys()
        else: # training vocab
            for file in tqdm(files):
                with open(f"{self.articles_dir}/{file}", 'r') as f:
                    try:
                        articles = json.load(f)
                        for article in articles.values():
                            yield article['text'] 
                    except json.decoder.JSONDecodeError:
                        pass

    def construct(self, data): # some months shouldn't exist bcs ess rounds
        X, W = torch.zeros((31, 1000, self.sample_size)), torch.zeros((31, 1000))
        Y = torch.tensor([data['values']['mean'], data['values']['var']])
        with open(f"{self.articles_dir}/{data['lang']}.json", 'r') as f:
            articles = json.load(f)
        for idx, (k1, v1) in enumerate(data['dailies'].items()):
            for jdx, (k2, v2) in enumerate(v1.items()):
                if k2 in articles:
                    text = articles[k2]['text']
                    tokens = self.tokenizer.vectorize(text)
                    X[idx, jdx, :] = tokens
                    W[idx, jdx] = v2
        return X, W, Y

    def __iter__(self):
        return self.get_stream(self.article_files)


# dev stack
def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

