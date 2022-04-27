# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src.utils import variables
from src.ess import ESS

import torch
from torch import nn, tensor
import torch.nn.functional as F

import gensim
import fasttext
import numpy as np

from collections import defaultdict
import random
import json
import time


# virian dataset
class Dataset(torch.utils.data.Dataset):

    data_dir = variables['data_dir']

    def __init__(self, params):
        self.params = params
        self.langs  = params['Languages']
        self.embs   = self._load_embs(params) # make dict of embs for langs
        self.days   = self._load_days(self.langs)
        self.keys   = list(self.days.keys()) # ["da_2020_10_30", ..., "..."]
        self.ess    = ESS() 
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        lang = self.keys[idx][:2]
        date = self.keys[idx][3:]
        return self.construct(lang, date, self.days[self.keys[idx]])
        
    def k_fold(self, lang):
        val_idx   = [idx for idx, sample in enumerate(self.keys) if sample[:2] == lang] # TODO: ++ val langs
        train_idx = [idx for idx in range(len(self.keys)) if idx not in val_idx]
        return train_idx, val_idx

    def construct(self, lang, date, days_text):
        X = self._titles_to_tensor(lang, days_text)
        W = tensor([text['views'] for text in days_text])
        W = F.pad(W, pad=(0,1000-W.shape[0])) / torch.max(W)
        # W = W[None, None, :, :]
        Y = self.ess.get_target(lang, date)
        return X, W, Y

    def _titles_to_tensor(self, lang, texts):
        X = tensor([self.embs[lang][title] for title in [t['article'] for t in texts]])
        X = F.pad(X, value=self.params['Vocab Size'], pad=(0,0,0,1000 - X.shape[0]))
        # consider averaging vectors in sentence to disregard order
        return X

    def _load_embs(self, params):
        embs = {}
        for lang in params['Languages']:
            with open(f"{self.data_dir}/wiki/embs_{lang}_{self.params['Vocab Size']}.json", 'r') as f:
                embs[lang] = defaultdict(lambda: [0 for _ in range(params['Embedding Dim'])], json.load(f)['texts'])
        return embs


    def _load_toks(self, langs):
        toks = {}
        for lang in langs:
            with open(f"{self.data_dir}/wiki/toks_{lang}_{self.params['Vocab Size']}.json", 'r') as f:
                toks[lang] = defaultdict(lambda: [self.params['Vocab Size'] for _ in range(self.params["Sample Size"])], json.load(f))
        return toks

    def _load_days(self, langs):
        days = {}
        for lang in langs:
            with open(f"{self.data_dir}/wiki/days_{lang}.json", 'r') as f:
                for date, days_text in json.load(f).items():
                    days[f"{lang}_{date}"] = days_text
        return days

    def _extend(self, sample):
        return sample + [self.params['Vocab Size'] for _ in range(self.params["Sample Size"] - len(sample))]


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

