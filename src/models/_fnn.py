# fnn.py
# by: Noah Syrkis
#   virian model

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# feedforwrd neural net
class FNN(nn.Module):

    def __init__(self, params):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(50, 50)

    def forward(self, x, w):
        z = self.encode(x)
        y = self.infer(z, w)
        x = self.decode(z)
        return x, y

    def encode(self, x):
        z = x
        return z

    def decode(self, z):
        x = z
        return x

    def inter(self, z, w):
        y = z
        return y


# dev calls
def main():
    from dataset import Dataset
    # ds = Dataset(p, params)

if __name__ == '__main__':
    main()

