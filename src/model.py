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
        self.drop   = nn.Dropout()

        self.weight   = nn.Parameter(torch.rand(1000))
        self.fc_inf_1 = nn.Linear(1000, 400)
        self.fc_inf_2 = nn.Linear(400, 21)

        self.fc_enc_1 = nn.Linear(300, 150)
        self.fc_enc_2 = nn.Linear(150, 150)
        self.fc_enc_3 = nn.Linear(150, 2)

        self.fc_dec_1 = nn.Linear(2, 150)
        self.fc_dec_2 = nn.Linear(150, 150)
        self.fc_dec_3 = nn.Linear(150, 300)

    def forward(self, x, w):
        z = self.encode(x)
        y = self.infer(z, w)
        x = self.decode(z)
        return x, y

    def infer(self, z, w):
        # w = self.weigh(w) # how should views be weighed?
        w = self.drop(w)  # drop half of all articles
        z = z * w         # weight articles by views
        z = self.fc_inf_1(z.mT)
        z = F.relu(z)
        z = self.fc_inf_2(z)
        z = torch.tanh(z)
        return z

    def encode(self, x): # 1000 x 300 -> 1000 x 2
        x = self.fc_enc_1(x)
        x = F.relu(x)
        x = self.fc_enc_2(x)
        x = F.relu(x)
        xx = self.drop(x)
        x = self.fc_enc_3(x)
        x = F.relu(x)
        return x

    def decode(self, z): # 1000 x 2 -> 1000 x 300
        z = self.fc_dec_1(z)
        z = F.relu(z)
        z = self.fc_dec_2(z)
        z = F.relu(z)
        z = self.drop(z)
        z = self.fc_dec_3(z)
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

