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
        self.encoder = Encoder(params) 
        self.infer   = Infer(params)
        self.decoder = Decoder(params) 

    def reparam(self, mu, sigma):
        sigma = torch.exp(sigma / 2)  # sigma or var?
        eps   = torch.randn_like(sigma) # why?
        return mu + sigma * eps
    
    def forward(self, x, w):
        mu, sigma = self.encoder(x)
        z         = self.reparam(mu, sigma)
        # y         = self.infer(z, w)
        x         = self.decoder(z)
        return x


# compress text into "themes"
class Encoder(nn.Module):
    """
    gets a batch, though it does not matter that
    articles are from different batches, so we flatten.
    For infering Y (ESS) we need to reshape.
    """
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.conv1  = nn.Conv2d(1, 3, 4) # one color -> 3 color (kernel size 4)
        self.conv2  = nn.Conv2d(3, 1, 3) # 3 color -> 1 color (kernel size 3)
        self.fc1    = nn.Linear(18, 10)  # reduce to 10 d
        self.fc2    = nn.Linear(18, 10)  # reduce to 10 d
        self.pool   = nn.MaxPool2d(4)    # kernel size 4
        self.drop   = nn.Dropout(0.5)    # dropout .7

    def forward(self, x):
        x     = x.reshape(-1, 1, x.shape[-2], x.shape[-1]) # debatch
        x     = self.pool(F.relu(self.conv1(x)))
        x     = self.pool(F.relu(self.conv2(x)))
        x     = x.reshape(-1, 18) # remove empty dimensions
        print(x.shape)
        exit()
        mu    = self.drop(self.fc1(x))
        sigma = self.drop(self.fc2(x))
        return mu, sigma


# decode text from "themes"
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.conv1  = nn.Conv2d(1, 3, 4) # one color -> 3 color (kernel size 4)
        self.conv2  = nn.Conv2d(3, 1, 3) # 3 color -> 1 color (kernel size 3)
        self.fc1    = nn.Linear(10, 300)
        self.drop   = nn.Dropout(0.5)

    def forward(self, z):
        z = self.drop(F.relu(self.fc1(z)))
        z = z.reshape(1, 1, z.shape[-1])
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.tanh(z)
        z = z.reshape(-1, 1000, self.params['Sample Size'], self.params['Embedding Dim'])
        return z


# infer ess from compressed text
class Infer(nn.Module):
    def __init__(self, params):
        super(Infer, self).__init__()
        self.weight = nn.Linear(1000, 1000)
        self.conv1  = nn.Conv2d(1, 1, 4)
        self.pool1  = nn.MaxPool2d(4)
        self.fc1    = nn.Linear(249, 10)
         

    def forward(self, z, w):
        w = self.weight(w)
        w = w.reshape(*w.shape, 1)
        z = z.reshape(-1, 1000, 10) # rebatch
        z = torch.mul(z, w)
        z = z.reshape(z.shape[0], 1, z.shape[1], z.shape[2])
        z = self.conv1(z)
        z = self.pool1(z)
        z = z.reshape(z.shape[0], z.shape[2])
        z = self.fc1(z)
        z = z.reshape(z.shape[0], 2, 5)
        return z
        

# dev calls
def main():
    from dataset import Dataset
    # ds = Dataset(p, params)

if __name__ == '__main__':
    main()

