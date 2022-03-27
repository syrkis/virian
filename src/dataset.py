# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
import os
import json
import torch
from itertools import cycle
from src.utils import get_s3, load
from tqdm import tqdm
from hashlib import sha256


# wiki dataset
class Dataset(torch.utils.data.IterableDataset):

    vocab_size  = 2 ** 16  # TODO: multilingual vocab
    sample_size = 2 ** 7   # 128 word wiki summaries
    s3          = get_s3() # AWS for remote storage

    def __init__(self, tokenizer=None):
        self.texts     = load('texts') # wikipedia summaries
        self.days      = load('days')  # wikipedia dailies
        self.ess       = load('ess')   # ess factors
        self.tokenizer = tokenizer
        
    def process_data(self):
        if self.tokenizer:
            for lang, data in self.days.items():
                yield self.construct(data, lang)
        else:
            for title, text in self.texts.items():
                yield text

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

    def get_stream(self):
        return cycle(self.process_data())

    def __iter__(self):
        return self.get_stream()


# dev stack
def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

