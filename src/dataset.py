# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from torch.utils.data import Dataset, IterableDataset
from itertools import cycle
from src.helpers import get_s3
import pandas as pd
import os, json, random


# wiki dataset
class Dataset(IterableDataset):

    vocab_size = 2 ** 12
    months_dir = "../data/months"
    articles_dir = "../data/articles"
    monthly_files = [f for f in os.listdir(months_dir) if f[-5:] == ".json"]
    article_files = [f for f in os.listdir(articles_dir) if f[-5:] == ".json"]
    # s3 = get_s3()
    # remote = s3.get_object(Bucket="data", Key="wiki/20200301.en.100k")

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        
    def get_stream(self, files):
        return cycle(self.process_data(files))

    def process_data(self, files):
        if self.tokenizer: # training model
            for file in files:
                with open(f"{self.months_dir}/{file}", 'r') as f:
                    yield self.construct(f.read())
        else: # training vocab
            for file in files:
                with open(f"{self.articles_dir}/{file}", 'r') as f:
                    articles = json.load(f)
                    for article in articles.values():
                        yield article['text'] 

    def construct(self, month):
        data = json.loads(month)
        X, W, Y, texts = torch.tensor(), torch.tensor(), torch.tensor(), []
        with open(f"{self.articles_dir}/{data['lang']}.json", 'r') as f:
            articles = json.load(f)
        for k1, v1 in data['dailies'].items():
            for k2, v2 in v1.items():
                if k2 in articles:
                    if self.tokenizer:
                        pass
                    else:
                        texts.append(articles[k2]['text'])
        return X, W, Y if self.tokenizer else texts # for model training or vocab training

    def __iter__(self):
        return self.get_stream(self.monthly_files)


# dev stack
def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

