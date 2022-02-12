# model.py
#   virian model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn


# topic model
class TopicModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.enc = nn.Linear(vocab_size, 50)
        self.dec = nn.Linear(50, vocab_size)
        self.ess = nn.Linear(50, 21) # ess human value questions

    def forward(self, x):
        y = self.enc(x)
        x = self.dec(y)
        y = self.ess(y)
        return x, y


# ess value model
class ValueModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 21)

    def forward(self, x):
        x = self.fc1(x)
        return x


def main():
    from dataset import Dataset
    ds = Dataset()
    print(ds[0])

if __name__ == '__main__':
    main()
