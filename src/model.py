# model.py
#   virian model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# topic model
class Model(nn.Module):

    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.enc    = nn.Linear(self.params["Embedding Dim"], 50)
        self.fc1    = nn.Linear(50, 10)
        self.dec    = nn.Linear(50, self.params["Embedding Dim"])

    def forward(self, x, w):
        x = self.encode(x)
        y = self.infer(x, w)
        x = self.decode(x)
        return x, y

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, x):
        x = self.dec(x)
        return x

    def infer(self, x, w):
        y = self.fc1(x) 
        y = y * w[:,:,None,None]
        y = torch.sum(y, dim=1) # collapse 1000 summaries
        y = torch.sum(y, dim=1) # collapse 16 words
        y = y.reshape((self.params["Batch Size"], 2, 5))
        return y


# dev calls
def main():
    from dataset import Dataset
    ds = Dataset()
    print(ds[0])

if __name__ == '__main__':
    main()
