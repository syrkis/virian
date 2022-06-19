# tnn.py
# by: Noah Syrkis
#   virian model

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# transformer neural net
class TNN(nn.Module):

    def __init__(self, params):
        super(CNN, self).__init__()
        self.n     = 10 ** 3
        self.emb   = params["Embedding Dim"]
        self.kern  = 10 # divisible twice by 1000 and 300
        self.pool  = nn.MaxPool2d(self.kern, padding=self.kern//2)
        self.drop  = nn.Dropout(0.5)

        # encode
        self.conv1 = nn.Conv2d(1, 3, self.kern)
        self.conv2 = nn.Conv2d(3, 1, self.kern)
        self.fc0   = nn.Linear(1000, 1000)
        self.fc1   = nn.Linear(self.emb // self.kern ** 2, self.emb // self.kern ** 2)

        # decode
        self.fc2   = nn.Linear(self.emb // self.kern ** 2, self.emb // self.kern)
        self.fc3   = nn.Linear(self.emb // self.kern, self.emb)
        self.fc4   = nn.Linear(self.n // self.kern ** 2, self.n // self.kern)
        self.fc5   = nn.Linear(self.n // self.kern, self.n)

        # infer
        self.fc6   = nn.Linear(self.emb // self.kern ** 2, 21) # esss cols (or facts) (make variable)
        self.fc8   = nn.Linear(self.n // self.kern ** 2, 2) # average nd variance

    def encode(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[-2], x.shape[-1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], x.shape[-2], x.shape[-1])
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return x

    def decode(self, z):
        z = F.relu(self.fc2(z))
        z = self.drop(z)
        z = F.relu(self.fc3(z))
        z = z.reshape(z.shape[0], z.shape[2], -1)
        z = F.relu(self.fc4(z))
        z = torch.tanh(self.fc5(z))
        z = z.reshape(z.shape[0], z.shape[2], -1)
        # z = F.normalize(z)
        return z

    def infer(self, z, w):
        w = self.fc0(w)[:, :, None]
        x = torch.mul(x, w)
        z = F.relu(self.fc6(z))
        z = self.drop(z)
        z = z.reshape(z.shape[0], z.shape[-1], -1)
        z = torch.sigmoid(self.fc7(z))
        return z

    def forward(self, x, w):
        z = self.encode(x)
        y = self.infer(z, w)
        x = self.decode(z)
        return x, y


# dev calls
def main():
    from dataset import Dataset
    # ds = Dataset(p, params)

if __name__ == '__main__':
    main()

