# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src.utils import *
import torch
import torch.nn.functional as F
import random


# virian dataset
class Dataset(torch.utils.data.Dataset):

    vocab_size  = hypers['vocab_size']  # TODO: multilingual vocab?
    sample_size = hypers['sample_size'] # 128 word wiki summaries
    embedder    = get_embeddings()

    def __init__(self, langs):
        self.langs = langs                  # for training or test?
        self.ess   = get_ess()              # ess factors
        self.toks  = load('toks')           # wiki summaries
        self.days  = load('days', langs)    # wiki dailies
        self.keys  = list(self.days.keys()) # day keys
        random.shuffle(self.keys)           # day keys

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        return self.construct(self.keys[idx], self.days[self.keys[idx]])
        
    def construct(self, meta, data):
        A = [title_hash(a['article']) for a in data['data']] 
        X = torch.tensor([self.toks[meta[:2]][t] for t in A])
        X = self.embedder(F.pad(X, pad=(0,0,0,1000 - len(A))).int())
        W = torch.tensor([a['views'] for a in data['data']])
        W = F.pad(W, pad=(0,1000-W.shape[0])) / torch.sum(W)
        Y = month_to_ess(meta[:2], meta[3:], self.ess)
        return X, W, Y


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

