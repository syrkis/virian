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
    sample_size = 2 ** 8
    months_dir = "../data/months"
    dailies_dir = "../data/dailies"
    articles_dir = "../data/articles"
    articles = {}
    for file in os.listdir(articles_dir):
        if file[-5:] == '.json':
            with open(f"{articles_dir}/{file}", 'r') as f:
                articles[file[:2]] = json.load(f)
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
        raw_X = []
        raw_W = []
        for article in articles:
            title_hash = sha256((article['article']).encode('utf-8')).hexdigest()
            if title_hash in self.articles[lang]:
                article_text = self.articles[lang][title_hash]['text']
                tokens = self.tokenizer.vectorize(article_text)
                raw_X.append(tokens)
            else:
                raw_X.append(torch.zeros(self.sample_size))
            raw_W.append(article['views']) 

        return raw_X
        """
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
        """

    def __iter__(self):
        return self.get_stream(self.article_files)


# dev stack
def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

