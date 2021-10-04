# datasets.py
#   virian wiki summary datasets
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import linecache
from transformers import AutoModelForMaskedLM, DistilBertTokenizerFast
import os
import json
import wikipedia

# wiki summary dataset
class Dataset(Dataset):

    # run on class instanciation
    def __init__(self, local=True):
    
        # if True wiki dump, else days
        self.local = local

        # dataset size (for wiki dump)
        self.size = 2 ** 10 # 5_308_416

        # off-line wiki dailies
        self.data_dir = '../data'

        # bert version
        self.lm = 'distilbert-base-uncased'

        # tokenizer                 
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.lm)

        # tokenize function
        self.tokenize = lambda x: self.tokenizer.batch_encode_plus(x)['input_ids']

        # model
        self.model = AutoModelForMaskedLM.from_pretrained(self.lm)

        # idf array
        self.idf = torch.load('../models/idf.pt')

        # we're using train data
        if local:
           
            self.data = []
            with open(f"{self.data_dir}/tok.csv", 'r') as f:
                for i in range(self.size):
                    line = list(map(int, f.readline().strip().split(',')))
                    # tmp = [0 for _ in range(512 - len(line))] # padding useless because 1d vec representation
                    # tmp.extend(line)
                    line = torch.tensor(line)
                    self.data.append(line)
 
        else:
            targets = set()
            days = [target for target in os.listdir(self.days) if target[-4:] == 'json']
            for day in days:
                articles = [article['article'] for article in json.load(open(f"{self.data_dir}/wikipedia/{day}", 'r'))]
                for article in articles:
                    targets.add(article)
            self.data = list(targets)

    # defining the concept of dataset length
    def __len__(self):
        return self.size

    # def fine what dataset[idx] returns
    def __getitem__(self, idx):
        if self.local:
            sample = self.data[idx]
        else:
            summary = wikipedia.summary(self.data[idx])
            sample = torch.tensor(self.tokenizer.encode_plus(summary)['input_ids'])
        w = self.tf_idf(self.tf(sample), sample)
        word_embeds = self.embed(sample)
        return torch.sum(w * word_embeds, dim=0)

    def tf_idf(self, tf, sample):
        tf_idf = tf * self.idf 
        w = tf_idf[sample]
        w /= torch.sum(w)
        return w[:, None]

    # term freq sample calculator     
    def tf(self, tokens):
        return torch.bincount(tokens, minlength=self.tokenizer.vocab_size)

    # idf calcualtor
    def idf(self):
        w = torch.ones(self.tokenizer.vocab_size)
        with open('../data/tok.csv', 'r') as f: 
            for _ in tqdm(range(self.size)):
                sample = list(map(int, f.readline().strip().split(',')))
                sample.append(0)
                w[list(set(sample))] += 1
        w = torch.log(self.size / w)
        torch.save(w, '../models/idf.pt')

    # convert data to doc embeddings
    def embed(self, tokens):
        return self.model.distilbert.embeddings.word_embeddings.weight[tokens]


# dev calls
def main():
    from tqdm import tqdm
    ds = Dataset(device, 10 ** 3)
    for i in range(1):
        print(ds[i])
    # loader = DataLoader(dataset=ds, batch_size=2 ** 12)
    # idf(loader, ds, torch.ones(ds.tokenizer.vocab_size).to(ds.device))

if __name__ == '__main__':
    main()

