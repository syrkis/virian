# model.py
#   virian inference model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# define the model
class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.weight = nn.Parameter(torch.rand(1000))
        self.drop   = nn.Dropout()
        self.fc1    = nn.Linear(300, 2)
        self.fc2    = nn.Linear(2, 300)
        self.fc3    = nn.Linear(1000, 21)

    def forward(self, x, w):
        z = self.encode(x)
        y = self.infer(z, w)
        x = self.decode(z)
        return x, y

    def infer(self, z, w):
        w = self.weigh(w) # how should views be weighed?
        z = z * w         # weight articles by views
        z = self.fc3(z.mT)
        z = torch.tanh(z)
        return z

    def encode(self, x): # 1000 x 300 -> 1000 x 2
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        return x

    def decode(self, z):
        z = self.fc2(z)
        z = torch.tanh(z)
        return z

    def weigh(self, w):
        w = w * self.weight
        w = w[:, :, None]
        return w


# call the model
def main():
    pass

if __name__ == '__main__':
    main()

