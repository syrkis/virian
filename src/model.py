# models.py
#   virian nlp models
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# document embedding model
class Model(nn.Module):

    # init call
    def __init__(self, embed_dim):

        self.comp_dim = embed_dim # haidth dimension count

        self.embed_dim = embed_dim # distilbert embed dim
          
        # super class initialization
        super(Model, self).__init__()   

        # compression / encoding layer
        self.enc = nn.Linear(self.embed_dim, self.comp_dim)

        # decompression / deconding layer
        self.dec = nn.Linear(self.comp_dim, self.embed_dim)

    # forward pass 
    def forward(self, x):
        # c = self.enc(x)
        # x = self.dec(c)
        return x

# dev calls
def main():
    from torch.utils.data import DataLoader
    from dataset import Dataset
    from tqdm import tqdm

    ds = Dataset()
    embed_size = ds.model.distilbert.embeddings.word_embeddings.weight.shape[1]
    model = Model(embed_size)
    loader = DataLoader(dataset=ds, batch_size=32)
    for batch in loader:
        x = model(batch)
        var = torch.var(x, dim=0).detach().numpy()
        avg = torch.mean(x, dim=0).detach().numpy()
        print(np.mean(np.mean(var)))

if __name__ == '__main__':
    main()
