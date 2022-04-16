# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src.utils import variables, parameters
from src.ess import ESS

import torch
from torch import nn, tensor
import torch.nn.functional as F
from bpemb import BPEmb

from collections import defaultdict
import random
import json


# virian dataset
class Dataset(torch.utils.data.Dataset):

    vocab_size    = parameters['vocab_size']
    embedding_dim = parameters['embedding_dim']
    sample_size   = parameters['sample_size']
    data_dir      = variables['data_dir']
    pad           = variables['pad']

    def __init__(self, langs):
        self.emb   = self._load_emb()
        self.langs = langs
        self.ess   = ESS() 
        self.days  = self._load_days(self.langs)
        self.toks  = self._load_toks(self.langs)
        self.keys  = list(self.days.keys()) # ["da_2020_10_30", ..., "..."]
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        lang = self.keys[idx][:2]
        date = self.keys[idx][3:]
        return self._construct(lang, date, self.days[self.keys[idx]])
        
    def k_fold(self, lang):
        val_idx   = [idx for idx, sample in enumerate(self.keys) if sample[:2] == lang] # TODO: ++ val langs
        train_idx = [idx for idx in range(len(self.keys)) if idx not in val_idx]
        return train_idx, val_idx

    def _construct(self, lang, date, days_text):
        X = [text['article'] for text in days_text] 
        X = [self.toks[lang][title][:self.sample_size] for title in X]
        X = tensor([self._extend(article) for article in X])
        X = self.emb(F.pad(X, value=self.pad, pad=(0,0,0,1000 - X.shape[0]))) # final X
        W = tensor([text['views'] for text in days_text])
        W = F.pad(W, pad=(0,1000-W.shape[0])) / torch.max(W)                  # final W
        Y = self.ess.get_target(lang, date)                                   # final Y
        return X, W, Y

    def _load_emb(self):
        emb = BPEmb(lang="multi", vs=self.vocab_size, dim=self.embedding_dim, add_pad_emb=True)
        emb = nn.Embedding.from_pretrained(tensor(emb.vectors), padding_idx=self.pad)
        return emb
        
    def _load_toks(self, langs):
        toks = {}
        for lang in langs:
            with open(f"{self.data_dir}/wiki/toks_{lang}.json", 'r') as f:
                toks[lang] = defaultdict(lambda: [self.pad for _ in range(self.sample_size)], json.load(f))
        return toks

    def _load_days(self, langs):
        days = {}
        for lang in langs:
            with open(f"{self.data_dir}/wiki/days_{lang}.json", 'r') as f:
                for day in f:
                    days[f"{lang}_{json.loads(day)['date']}"] = json.loads(day)['data']
        return days

    def _extend(self, sample):
        return sample + [self.pad for _ in range(self.sample_size - len(sample))]


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

