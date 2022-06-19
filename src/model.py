# model.py
#   virian inference model
# by: Noah Syrkis

# imports
from torch import nn


# define the model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.queueSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, w):
        z = self.encode(x)
        y = self.infer(z, w)
        x = self.decode(z)
        return x, y

    def infer(self, z, w):
        return z

    def encode(self, x):
        return queue


    def decode(self, z):
        return z


# call the model
def main():
    pass

if __name__ == '__main__':
    main()

