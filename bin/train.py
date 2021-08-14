# train.py training virian word embeddings
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from tqdm import tqdm


# train function
def train(model, loader, n_epochs, window, optimizer, criterion):
    for epoch in range(n_epochs):
        for batch in tqdm(loader):
            optimizer.zero_grad() 
            loss = 0
            for i in range(window, batch.shape[1] - (window + 1)):
                x = batch[:, [j for j in range(i - window, i + window + 1) if j != i]]
                y = batch[:, i].flatten()
                pred = model(x)
                print(pred, y)
                loss += criterion(pred, y)
            loss.backward()
            optimizer.step()


# dev stack
def main():
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    ds = Dataset()
    n_epochs = 1
    window = 5
    loader = DataLoader(dataset=ds, batch_size=16, shuffle=True) 
    train(model, loader, n_epochs, window, optimizer, criterion)
    
if __name__ == '__main__':
    main()
