# model.py
#   virian model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import hypers


# topic model
class Model(nn.Module):

    sample_size = hypers['sample_size']
    embedding_dim = hypers['embedding_dim']

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(self.embedding_dim, 50)
        self.fc1 = nn.Linear(50, 10)
        self.dec = nn.Linear(50, self.embedding_dim)

    def forward(self, x, w, y):
        x = self.encode(x)
        # y = self.infer(x, w)
        x = self.decode(x)
        return x # , y

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, x):
        x = self.dec(x)
        return x

    def infer(self, x, w):
        y = self.fc1(x) 
        # y = w[:,:,None] * y
        # y = torch.sum(y, dim=1)
        # y = y.reshape(x.shape[0], 5, 2)
        return y


# dev calls
def main():
    from dataset import Dataset
    ds = Dataset()
    print(ds[0])

if __name__ == '__main__':
    main()
