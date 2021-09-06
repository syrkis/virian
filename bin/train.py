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
import mlflow
from tqdm import tqdm


# document dimension reduction trainer
def train(model, loader, n_epochs, optimizer, criterion, device):

    # for every epoch
    for epoch in range(n_epochs):

        # run through all batches
        for batch in tqdm(loader):

            # but batch on device
            batch.to(device)

            # clear gradient
            optimizer.zero_grad()

            # make prediction (_ contains compressed 5d batch representations)
            pred, _ = model(batch)

            # calcualte loss
            loss = criterion(pred, batch)   

            # back propagate
            loss.backward()

            # improve parameters
            optimiser.step()


# dev stack
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    ds = Dataset(device, None, 10 * 5)
    embed_dim = ds.model.distilbert.embeddings.word_embeddings.weight.shape[1]
    vocab_size = ds.tokenizer.vocab_size
    model = (Model(vocab_size, embed_dim)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 1
    loader = DataLoader(dataset=ds, batch_size=16, shuffle=True) 
    train(model, loader, n_epochs, optimizer, criterion, device)
    
if __name__ == '__main__':
    main()
