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
        self.encoder = Encoder(params) 
        self.infer   = Infer(params)
        self.decoder = Decoder(params) 

    def forward(self, x, w):
        z = self.encoder(x)
        y = self.infer(z, w)
        x = self.decoder(z)
        return x, y


# compress text into "themes"
class Encoder(nn.Module):
    """
    gets a batch, though it does not matter that
    articles are from different batches, so we flatten.
    For infering Y (ESS) we need to reshape.
    """
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 4)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(3, 1, 3)
        self.pool2 = nn.MaxPool2d(4)
        self.drop  = nn.Dropout(0.7)
        self.fc1   = nn.Linear(18, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.reshape(-1, 18) # floor(300 / 4 / 4)
        x = self.fc1(x)
        x = self.drop(x)
        return x


# decode text from "themes"
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        # layers
        self.fc1  = nn.Linear(10, 300)
        self.fc2  = nn.Linear(300, 300 * params['Sample Size'])
        self.drop = nn.Dropout(0.7)

        # parameters
        self.sample_size = params['Sample Size']
        self.emb_dim     = params['Embedding Dim']

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.drop(z)
        z = F.relu(self.fc2(z))
        z = z.reshape(-1, 1000, self.sample_size, self.emb_dim)
        return z


# infer ess from compressed text
class Infer(nn.Module):
    def __init__(self, params):
        super(Infer, self).__init__()
        self.weight = nn.Linear(1000, 1000)
        self.conv1 = nn.Conv2d(1, 1, 4)
        self.pool1 = nn.MaxPool2d(4)
        self.fc1   = nn.Linear(249, 10)
         

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

