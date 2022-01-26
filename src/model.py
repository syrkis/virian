# model.py
#   virian model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.enc = nn.Linear(vocab_size, 50)
        self.dec = nn.Linear(50, vocab_size)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


def main():
    from dataset import Dataset
    ds = Dataset()
    print(ds[0])

if __name__ == '__main__':
    main()
