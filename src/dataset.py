# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src.utils import *
from src.ess import ESS
import torch
from torch import nn, tensor
import torch.nn.functional as F
import random
from bpemb import BPEmb
from collections import defaultdict


# virian dataset
class Dataset(torch.utils.data.Dataset):

    vocab_size  = hypers['vocab_size']  # TODO: multilingual vocab?
    sample_size = hypers['sample_size'] # 128 word wiki summaries
    pad         = 10 ** 6

    def __init__(self, langs):
        self.emb             = self._load_emb()
        self.langs           = langs
        self.ess             = ESS() 
        self.days, self.toks = self._load_days_and_toks()
        self.keys            = list(self.days.keys())
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        lang = self.keys[idx][:2]
        date = self.keys[idx][3:]
        return self.construct(lang, date, self.days[self.keys[idx]])
        
    def construct(self, lang, date, days_text):
        T = [text['article'] for text in days_text] 
        X = [self.toks[lang][title][:self.sample_size] for title in T]
        X = tensor([self._extend(article) for article in X])
        X = F.pad(X, value=self.pad, pad=(0,0,0,1000 - len(T)))
        X = self.emb(X)
        W = tensor([text['views'] for text in days_text])
        W = F.pad(W, pad=(0,1000-W.shape[0])) / torch.sum(W)
        Y = self.ess.get_target(lang, date)
        return X, W, Y

    def k_fold(self, lang):
        val_idx   = [idx for idx, sample in enumerate(self.keys) if sample[:2] == lang] # TODO: ++ val langs
        train_idx = [idx for idx in range(len(self.keys)) if idx not in val_idx]
        return train_idx, val_idx

    def _load_emb(self):
        emb = BPEmb(lang="multi", vs=self.pad, dim=300, add_pad_emb=True)
        emb = nn.Embedding.from_pretrained(tensor(emb.vectors), padding_idx=self.pad) # no padding
        return emb
        
    def _load_days_and_toks(self):
        days = self._load_days(self.langs)
        toks = {lang: self._load_toks(lang) for lang in self.langs}
        return  days, toks

    def _load_days(self, langs, days={}):
        for lang in langs:
            with open(f"{paths['wiki']}/days_{lang}.json", 'r') as f:
                for day in f:
                    days[f"{lang}_{json.loads(day)['date']}"] = json.loads(day)['data']
        return days

    def _load_toks(self, lang):
        with open(f"{paths['wiki']}/toks_{lang}.json", 'r') as f:
            return defaultdict(lambda: [self.pad for _ in range(hypers['sample_size'])], json.load(f))

    def _extend(self, sample):
        return sample + [self.pad for _ in range(self.sample_size - len(sample))]


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

