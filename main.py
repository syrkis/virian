# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, train
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# call stack
def main():
    ds = Dataset()
    loader = DataLoader(dataset=ds, batch_size=2)
    model = Model(ds.vocab_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    train(loader, model, optimizer, criterion)

if __name__ == "__main__":
    main()
