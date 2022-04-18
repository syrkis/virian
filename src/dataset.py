# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src.utils import variables
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

    data_dir      = variables['data_dir']
    pad           = variables['pad']

    def __init__(self, langs, params):
        self.params = params
        self.emb    = self._load_emb()
        self.langs  = langs
        self.ess    = ESS() 
        self.days   = self._load_days(self.langs)
        self.toks   = self._load_toks(self.langs)
        self.keys   = list(self.days.keys()) # ["da_2020_10_30", ..., "..."]
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
        Y = self.ess.get_target(lang, date)
        return X, W, Y

    def _titles_to_tensor(self, lang, days_text):
        X = [text['article'] for text in days_text] 
        X = [self.toks[lang][title][:self.params["Sample Size"]] for title in X]
        X = tensor([self._extend(article) for article in X])
        X = self.emb(F.pad(X, value=self.pad, pad=(0,0,0,1000 - X.shape[0]))).detach()
        return X

    def _load_emb(self):
        emb = BPEmb(lang="multi", vs=self.params['Vocab Size'], dim=self.params['Embedding Dim'], add_pad_emb=True)
        emb = nn.Embedding.from_pretrained(tensor(emb.vectors), padding_idx=self.pad)
        return emb
        
    def _load_toks(self, langs):
        toks = {}
        for lang in langs:
            with open(f"{self.data_dir}/wiki/toks_{lang}.json", 'r') as f:
                toks[lang] = defaultdict(lambda: [self.pad for _ in range(self.params["Sample Size"])], json.load(f))
        return toks

    def _load_days(self, langs):
        days = {}
        for lang in langs:
            with open(f"{self.data_dir}/wiki/days_{lang}.json", 'r') as f:
                for date, days_text in json.load(f).items():
                    days[f"{lang}_{date}"] = days_text
        return days

    def _extend(self, sample):
        return sample + [self.pad for _ in range(self.params["Sample Size"] - len(sample))]


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

