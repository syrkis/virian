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
from transformers import AutoTokenizer, AutoModelForMaskedLM
import requests
import os
import json


# wiki summary dataset
class Dataset(Dataset):

    # run on class instanciation
    def __init__(self, date=None):

        # if not date use 5M dump
        self.date = date

        # off-line wiki dailies
        self.local = '../../api/data/scrape/wikipedia'

        # 5M dump
        self.dump = '../../data/raw.tar' 

        # bert version
        self.bert = 'distilbert-base-uncased'

        # tokenizer                 
        self.tokenizer = lambda x: AutoTokenizer.from_pretrained(self.model)(x)['input_ids']

        # model
        self.model = AutoModelForMaskedLM.from_pretrained(self.model)

        # idf array
        self.idf = torch.tensor(np.loadtxt('../models/idf.csv', delimiter='\n'))

        # if target is daily and local
        if self.date and self.date in [file.split('.')[0] for file in os.listdir(self.local)]:

            # do this
            daily = json.load(open(f"{self.local}/{self.date}.json", 'r'))

        # if target is daily and remote
        elif self.date:

            # do this
            daily = None

        # if we're looping through the train data
        else:

            # load in the huge dataset
            tokens = map(self.tokenizer, open(self.dump, 'r').readlines())

        # convert to doc embed
        self.data = self.embed(self, tokens)
        

    # defining the concept of dataset length
    def __len__(self):
        return self.data.shape[0]

    # def fine what dataset[idx] returns
    def __getitem__(self, idx):

        if self.date:
            if self.date in [file.split('.')[0] for file in os.listdir(self.local_days)]:
                res = json.load(f"{self.local_days}/{self.date}.json")
            else:
                res = requests.get("WIKIMEDIAAPI").json()

        # get tokenized summary for idx
        if not self.date and not self.doc:
            output = self.tokens(idx)

        # convert summary into docuemnt embedding  
        else:
            tokens = self.tokens(idx)
            tfidf = self.tf(tokens) * self.idf_array
            w = tfidf[tokens]
            w /= torch.sum(tokens)
            embeds = self.embed(tokens) 
            embeds *= w[:, None]
            output = torch.sum(embeds, dim=0)

        return output

    # term freq sample calculator     
    def tf(self, tokens):
        # zeros matrix for one tf freq counts, populate o and return vector
        o = torch.zeros((self.n_words, self.tokenizer.vocab_size)).to(self.device)
        o.scatter_(1, tokens.unsqueeze(1), 1)
        return torch.sum(o, dim=0)

    # idf calcualtor
    def idf(self, batch, idf):
        batch = batch.to(self.device)
        for sample in batch:
            tf = self.tf(sample)
            idf += (tf != 0).int()
        return idf


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
    loader = DataLoader(dataset=ds, batch_size=2 ** 12)
    # idf(loader, ds, torch.ones(ds.tokenizer.vocab_size).to(ds.device))

if __name__ == '__main__':
    main()

