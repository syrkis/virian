# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src.utils import *
from src.ess import ESS
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from bpemb import BPEmb
from collections import defaultdict


# virian dataset
class Dataset(torch.utils.data.Dataset):

    vocab_size  = hypers['vocab_size']  # TODO: multilingual vocab?
    sample_size = hypers['sample_size'] # 128 word wiki summaries

    def __init__(self, langs):
        self.langs           = langs
        self.ess             = ESS() 
        self.days, self.toks = self._load_days_and_toks()
        self.embed           = self._load_embed()
        self.keys            = list(self.days.keys())
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        lang = self.keys[idx][:2]
        date = self.keys[idx][3:]
        return self.construct(lang, date, self.days[self.keys[idx]])
        
    def construct(self, lang, date, days_articles):
        A = [title_hash(a['article']) for a in days_articles] 
        print(A)
        exit()
        X = torch.tensor([self.toks[lang][t] for t in A])
        X = self.embed(F.pad(X, value=-1, pad=(0,0,0,1000 - len(A))).int())
        W = torch.tensor([a['views'] for a in days_articles])
        W = F.pad(W, pad=(0,1000-W.shape[0])) / torch.sum(W)
        Y = self.ess.get_target(lang, date)
        return X, W, Y

    def k_fold(self, lang):
        val_idx   = [idx for idx, sample in enumerate(self.keys) if sample[:2] == lang] # TODO: multiple langs
        train_idx = [idx for idx in range(len(self.keys)) if idx not in val_idx]
        return train_idx, val_idx

    def _load_embed(self):
        embed = BPEmb(lang="multi", vs=10 ** 6, dim=300, add_pad_emb=True).vectors
        embed = nn.Embedding.from_pretrained(torch.tensor(embed))
        embed = embed.requires_grad_(requires_grad=False)
        return embed
        
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
            return defaultdict(lambda: [0 for _ in range(hypers['sample_size'])], json.load(f))


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

