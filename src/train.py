# trainers.py
#   training virian word embeddings
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from tqdm import tqdm


# document dimension reduction trainer
def train(model, loader, n_epochs, optimizer, criterion, device):

    # for every epoch
    for epoch in range(n_epochs):

        # run through all batches
        for batch in tqdm(loader):

            # but batch on device
            batch = batch.to(device)

            # clear gradient
            optimizer.zero_grad()

            # make prediction
            pred, _ = model(batch)

            # calcualte loss
            loss = criterion(pred, batch)   

            # back propagate
            loss.backward()

            # improve parameters
            optimizer.step()


# dev stack
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    ds = Dataset(device, None, 10 ** 5)
    embed_dim = ds.model.distilbert.embeddings.word_embeddings.weight.shape[1]
    model = Model(embed_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 1
    loader = DataLoader(dataset=ds, batch_size=2 ** 10, shuffle=True) 
    train(model, loader, n_epochs, optimizer, criterion, device)
    torch.save(model.state_dict(), 'models/model.pt')
    
if __name__ == '__main__':
    main()
