# model.py
#   virian model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
from src.utils import hypers


# topic model
class Model(nn.Module):

    sample_size = hypers['sample_size']

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(self.sample_size, 50)
        self.fc1 = nn.Linear(50, 10)
        self.dec = nn.Linear(50, self.sample_size)

    def forward(self, x, w, y):
        x = self.enc(x)
        y = torch.mean(self.fc1(x), dim = 1).reshape(x.shape[0], 5, 2)
        x = self.dec(x)
        return x, y


    def predict(self, x):
        pass

# dev calls
def main():
    from dataset import Dataset
    ds = Dataset()
    print(ds[0])

if __name__ == '__main__':
    main()
