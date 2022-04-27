# model.py
# by: Noah Syrkis
#   virian model

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# topic model
class Model(nn.Module):

    def __init__(self, params):
        super(Model, self).__init__()
        self.n     = 10 ** 3
        self.emb   = params["Embedding Dim"]
        self.kern  = 5 # divisible twice by 1000 and 300
        self.pool  = nn.MaxPool2d(self.kern, padding=self.kern//2) 
        self.drop  = nn.Dropout(0.6)

        # enc
        self.conv1 = nn.Conv2d(1, 3, self.kern)
        self.conv2 = nn.Conv2d(3, 1, self.kern)
        self.fc0   = nn.Linear(self.emb // self.kern ** 2, self.emb // self.kern ** 2)

        # dec
        self.fc1   = nn.Linear(self.n // self.kern ** 2, self.n // self.kern)
        self.fc2   = nn.Linear(self.n // self.kern, self.n)
        self.fc3   = nn.Linear(self.emb // self.kern ** 2, self.emb // self.kern)
        self.fc4   = nn.Linear(self.emb // self.kern, self.emb)

        # inf
        self.fc5   = nn.Linear(self.emb // self.kern ** 2, 2)
        self.fc6   = nn.Linear(self.n // self.kern ** 2, 5)

    def encode(self, x, w):
        x = x * w[:, :, None]
        x = x.reshape(x.shape[0], 1, x.shape[-2], x.shape[-1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], x.shape[-2], x.shape[-1])
        x = F.relu(self.fc0(x))
        return x

    def decode(self, z): # make 2self.kern times bigger
        z = z.reshape(z.shape[0], z.shape[2], -1) # get rid of reshape
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.reshape(z.shape[0], z.shape[2], -1)
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        return z

    def infer(self, z):
        z = F.relu(self.fc5(z))
        z = z.reshape(z.shape[0], z.shape[-1], -1)
        z = torch.tanh(self.fc6(z))
        return z
    
    def forward(self, x, w):
        z = self.encode(x, w)
        y = self.infer(z)
        x = self.decode(z)
        return x, y 


# dev calls
def main():
    from dataset import Dataset
    # ds = Dataset(p, params)

if __name__ == '__main__':
    main()

