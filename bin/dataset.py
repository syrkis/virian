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
from transformers import AutoTokenizer


# wiki summary dataset
class Dataset(Dataset):

    # run on class instanciation
    def __init__(self, live=False, doc=False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # tokenizer, sample count and word count
        self.doc = doc
        self.live = live
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.n_samples = 10 ** 3 # 5_315_384
        self.n_words = 2 ** 9

        # if doc centric
        if doc:
            self.embed = torch.load('../models/word_embed_100d.pt')
            self.idf = torch.tensor(np.loadtxt('../models/idf.csv', delimiter='\n'))

    # defining the concept of dataset length
    def __len__(self):
        return self.n_samples

    # def fine what dataset[idx] returns
    def __getitem__(self, idx):
        output = self.tokens(idx)
        if self.doc:
            tokens = self.tokens(idx)
            tfidf = self.tf(tokens) * self.idf
            w = tfidf[tokens]
            w /= torch.sum(tokens)
            embeds = self.embed(tokens) 
            embeds *= w[:, None]
            output = torch.sum(embeds, dim=0)
        return output

    # hepler function to tokenize and truncate / pad
    def tokens(self, idx):

        # load line and prep data file for quick access
        line = self._reader(idx)
        tokens = self.tokenizer(line)['input_ids']
        tokens = tokens[: min(self.n_words, len(tokens))]

        # if summary is to short
        if len(tokens) < self.n_words:
            tmp = [0 for _ in range(self.n_words - len(tokens))] 
            tmp.extend(tokens)
            tokens = tmp

        return torch.tensor(tokens)

    # get a given sample
    def _reader(self, idx):
        line = linecache.getline('../../data/raw.tar', idx)
        return line

    # term freq sample calculator     
    def tf(self, tokens):
        # zeros matrix for one tf freq counts, populate o and return vector
        o = torch.zeros((self.n_words, self.tokenizer.vocab_size)).to(self.device)
        o.scatter_(1, tokens.unsqueeze(1), 1)
        return torch.sum(o, dim=0)

    # idf calcualtor
    def tfidf(self, batch, idf):
        batch = batch.to(self.device)
        for sample in batch:
            tf = self.tf(sample)
            idf += (tf != 0).int()
        return idf


# idf constructer
def idf(loader, ds, idf):
    for batch in tqdm(loader):
        idf = ds.tfidf(batch, idf)
    idf = torch.log((len(loader.dataset) + ds.tokenizer.vocab_size) / idf)
    idf = idf.cpu().numpy()
    idf.tofile('../models/idf.csv', sep='\n')

# dev calls
def main():
    ds = Dataset()
    loader = DataLoader(dataset=ds, batch_size=32, shuffle=True)
    idf(loader, ds, torch.ones(ds.tokenizer.vocab_size).to(ds.device))

if __name__ == '__main__':
    main()

