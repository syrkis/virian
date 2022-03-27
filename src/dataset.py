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
from hashlib import sha256
import random


# wiki dataset
class Dataset(IterableDataset):

    vocab_size = 2 ** 16
    sample_size = 2 ** 7
    dailies_dir = "../data/wiki/days"
    articles_dir = "../data/wiki/text"
    articles = {}
    for file in os.listdir(articles_dir):
        if file[-5:] == '.json':
            with open(f"{articles_dir}/{file}", 'r') as f:
                articles[file[:2]] = json.load(f)
    article_files = [f for f in articles_dir if f[-5:] == '.json']
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
                    day = json.loads(f.readline())
                    yield self.construct(day, file[:2]) # file[:2] is lang (da)
        else: # training vocab
            for file in tqdm(files):
                with open(f"{self.articles_dir}/{file}", 'r') as f:
                    try:
                        articles = json.load(f)
                        for article in articles.values():
                            yield article['text'] 
                    except json.decoder.JSONDecodeError:
                        pass

    def construct(self, day, lang): # some months shouldn't exist bcs ess rounds
        date = day['date']
        articles = day['data'] 
        X = torch.zeros((1000, self.sample_size))
        W = torch.zeros(1000)
        Y = torch.zeros((5, 2))
        for idx, article in enumerate(articles):
            title_hash = sha256((article['article']).encode('utf-8')).hexdigest()
            if title_hash in self.articles[lang]:
                article_text = self.articles[lang][title_hash]['text']
                tokens = self.tokenizer.vectorize(article_text)
                X[idx] += tokens
            else:
                X[idx] += torch.zeros(self.sample_size)
            W[idx] += article['views']
        return X

    def __iter__(self):
        return self.get_stream(self.article_files)


# dev stack
def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

