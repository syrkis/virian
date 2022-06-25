# model.py
#   virian inference model
# by: Noah Syrkis

# imports
from torch import nn


# define the model
class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.weight = nn.Parameter(torch.rand(1000))

    def forward(self, x, w):
        print(x.shape, w.shape)
        z = self.encode(x)
        y = self.infer(z, w)
        x = self.decode(z)
        return x, y

    def infer(self, z, w): # make LSTM or attention
        w = w * self.wigh
        return z

    def encode(self, x):
        return x

    def decode(self, z):
        return z

# call the model
def main():
    pass

if __name__ == '__main__':
    main()

