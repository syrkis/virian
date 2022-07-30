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
from tqdm import tqdm
import random
import json
import time


# virian dataset
class Dataset(torch.utils.data.Dataset):

    data_dir = variables['data_dir']

    def __init__(self, conf, train=True):
        self.conf  = conf
        self.ess   = ESS(conf)
        if train:
            self.langs = conf['train_langs']
        else:
            self.langs = conf['test_langs']
        self.embs  = self.load_embs() # make dict of embs for lang
        self.days  = self._load_days(self.langs)
        self.keys  = self.filter_ranges(list(self.days.keys())) # lang_date
        random.shuffle(self.keys)
        random.shuffle(self.langs)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        lang = self.keys[idx][:2]
        date = self.keys[idx][3:]
        return self.construct(lang, date, self.days[self.keys[idx]])

    def filter_ranges(self, keys):
        out = []
        for key in keys:
            lang = key[:2]
            cntry = self.conf['langs'][lang]
            date = key[3:].replace('_', '-')
            for lo, hi in self.ess.ranges[cntry].values():
                if date <= hi and date >= lo:
                    out.append(key)
        return out

    def k_fold(self, langs):
        val_idx   = [idx for idx, sample in enumerate(self.keys) if sample[:2] in langs] # TODO: ++ val langs
        train_idx = [idx for idx in range(len(self.keys)) if idx not in val_idx]
        return train_idx, val_idx

    def construct(self, lang, date, texts):
        X = tensor([self.embs[lang][title] for title in [t['article'] for t in texts]])
        X = F.pad(X, value=0, pad=(0,0,0,1000 - X.shape[0]))
        W = tensor([text['views'] for text in texts])
        W = F.pad(W, pad=(0,1000-W.shape[0])) / torch.max(W)
        Y = self.ess.get_target(lang, date).T
        return X, W, Y

    def load_embs(self):
        embs = {}
        for lang in tqdm(self.langs):
            with open(f"{self.data_dir}/wiki/embs_1d_{lang}.json", 'r') as f:
                embs[lang] = defaultdict(lambda: [0] * 300, json.load(f)['texts'])
        return embs

    def _load_toks(self, langs): # from when samples were 1000 x 32 x 300 (BPEmb)
        toks = {}
        for lang in langs:
            with open(f"{self.data_dir}/wiki/toks_{lang}_{self.params['Vocab Size']}.json", 'r') as f:
                toks[lang] = defaultdict(lambda: [self.params['Vocab Size'] for _ in range(self.params["Sample Size"])], json.load(f))
        return toks

    def _load_days(self, langs):
        days = {}
        for lang in tqdm(langs):
            with open(f"{self.data_dir}/wiki/days_{lang}.json", 'r') as f:
                for date, days_text in json.load(f).items():
                    days[f"{lang}_{date}"] = days_text
        return days


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

