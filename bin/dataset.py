# datasets.py
#   virian wiki summary datasets
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import linecache
from transformers import AutoModelForMaskedLM, DistilBertTokenizerFast
import requests
import os
import json
import csv

# wiki summary dataset
class Dataset(Dataset):

    # run on class instanciation
    def __init__(self, date=None):

        # if not date use 5M dump
        self.date = date

        # off-line wiki dailies
        self.local = '../../api/data/scrape/wikipedia'

        # 5M dump
        self.dump = '../data/small.csv' 

        # bert version
        self.lm = 'distilbert-base-uncased'

        # tokenizer                 
        self.tokenizer = lambda x: DistilBertTokenizerFast.from_pretrained(self.lm).batch_encode_plus(x)['input_ids']

        # model
        self.model = AutoModelForMaskedLM.from_pretrained(self.lm)

        # idf array
        self.idf = torch.tensor(np.loadtxt('../models/idf.csv', delimiter='\n'))

        # if target is daily and local
        if self.date and self.date in [file.split('.')[0] for file in os.listdir(self.local)]:

            self.data = None

        # if target is daily and remote
        elif self.date:

            # do this
            daily = None
            self.data = None

        # we're using train data
        else:
           
            with open(self.dump, 'r') as f:
                self.data = [torch.tensor(list(map(int, line.split(',')))) for line in f.readlines()]
                
    # convert data to doc embeddings
    def embed(self, tokens):
        E = self.model.distilbert.embeddings.word_embeddings.weight
        return E[tokens]
        

    # defining the concept of dataset length
    def __len__(self):
        return self.data.shape[0]

    # def fine what dataset[idx] returns
    def __getitem__(self, idx):
        word_embed = self.embed(self.data[idx])
        print(word_embed)

        """
            tokens = self.tokens(idx)
            tfidf = self.tf(tokens) * self.idf_array
            w = tfidf[tokens]
            w /= torch.sum(tokens)
            embeds = self.embed(tokens) 
            embeds *= w[:, None]
            output = torch.sum(embeds, dim=0)
        """
        return 0

    # term freq sample calculator     
    def tf(self, tokens):
        # zeros matrix for one tf freq counts, populate o and return vector
        o = torch.zeros((self.n_words, self.tokenizer.vocab_size)).to(self.device)
        o.scatter_(1, tokens.unsqueeze(1), 1)
        return torch.sum(o, dim=0)

    """
    # idf calcualtor
    def idf(self, batch, idf):
        batch = batch.to(self.device)
        for sample in batch:
            tf = self.tf(sample)
            idf += (tf != 0).int()
        return idf
    """


# idf constructer
def idf(loader, ds, idf):
    for batch in tqdm(loader):
        idf = ds.idf(batch, idf)
    idf = torch.log((len(loader.dataset) + ds.tokenizer.vocab_size) / idf)
    idf = idf.cpu().numpy()
    idf.tofile('../models/idf.csv', sep='\n')

# dev calls
def main():
    ds = Dataset()
    for i in range(10):
        tmp = ds[i]
    # loader = DataLoader(dataset=ds, batch_size=2 ** 12)
    # idf(loader, ds, torch.ones(ds.tokenizer.vocab_size).to(ds.device))

if __name__ == '__main__':
    main()

