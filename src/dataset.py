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
    # s3 = get_s3()
    # remote = s3.get_object(Bucket="data", Key="wiki/20200301.en.100k")

    def __init__(self, tokenizer=None):
        self.files = [f for f in os.listdir(self.months_dir) if f[-5:] == ".json"]
        self.tokenizer = tokenizer
        
    def get_stream(self, files):
        return cycle(self.process_data(files))

    def process_data(self, files):
        for file in files:
            with open(f"{self.months_dir}/{file}", 'r') as f:
                yield self.construct(f.read())

    def construct(self, month):
        data = json.loads(month)
        with open(f"{self.articles_dir}/{data['lang']}.json", 'r') as f:
            articles = json.load(f)
        for daily in data.dailies:
            pass               
        return data.keys()

    def __iter__(self):
        return self.get_stream(self.files)


# dev stack
def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

