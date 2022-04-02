# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports from src import 
from src.utils import *
import torch
import torch.nn.functional as F
import random


# wiki dataset
class Dataset(torch.utils.data.Dataset):

    vocab_size  = hypers['vocab_size']  # TODO: multilingual vocab?
    sample_size = hypers['sample_size'] # 128 word wiki summaries
    tokenizer   = get_tokenizer()
    embedder    = get_embeddings()

    def __init__(self):
        self.ess  = get_ess()              # ess factors
        self.toks = load('toks')           # wiki summaries
        self.days = load('days')           # wiki dailies
        self.keys = list(self.days.keys()) # day keys
        random.shuffle(self.keys)          # day keys

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        return self.construct_sample(self.keys[idx], self.days[self.keys[idx]])
        
    def construct_sample(self, meta, data):
        titles   = [title_hash(article['article']) for article in data['data']] 
        toks     = torch.tensor([self.toks[meta[:2]][title] for title in titles])
        toks_pad = F.pad(toks, pad=(0,0,0,1000 - len(titles))).int()
        X        = self.embedder(toks_pad)
        views    = torch.tensor([article['views'] for article in data['data']])
        W        = F.pad(views, pad=(0,1000-views.shape[0])) / torch.sum(views)
        Y        = month_to_ess(meta[:2], meta[3:].replace('_', '/'), self.ess)
        return X, W, Y


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

