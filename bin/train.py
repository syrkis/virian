# train.py
#   training virian word embeddings
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import WordModel
from dataset import Dataset
from tqdm import tqdm


# train function
def train(model, loader, n_epochs, window, optimizer, criterion, idx=0):
    for epoch in range(n_epochs):
        for batch in tqdm(loader):
            optimizer.zero_grad() 
            loss = 0
            for i in range(window, batch.shape[1] - (window + 1)):
                x = batch[:, [j for j in range(i - window, i + window + 1) if j != i]]
                y = batch[:, i].flatten()
                pred = model(x)
                loss += criterion(pred, y)
            idx = metric(loss, batch, idx)
            loss.backward()
            optimizer.step()


def metric(loss, batch, idx):
    with open('loss.txt', 'a') as f:
        f.write(f"{idx},{round(loss.item() / batch.shape[0], 4)} \n")
    return idx + 1
    

# dev stack
def main():
    model = WordModel(30522, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    ds = Dataset()
    n_epochs = 1
    window = 10
    loader = DataLoader(dataset=ds, batch_size=16, shuffle=True) 
    train(model, loader, n_epochs, window, optimizer, criterion)
    torch.save(model.embed, '../models/word_embed_5d.pt')
    
if __name__ == '__main__':
    main()
