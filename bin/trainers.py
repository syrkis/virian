# trainers.py
#   training virian word embeddings
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import WordModel
from datasets import WordDataset
from tqdm import tqdm



# train standard non-contextual word embeddings
def word_embed_trainer(model, loader, n_epochs, window, optimizer, criterion, idx=0):

    # for every epoch
    for epoch in range(n_epochs):

        # run through all batches
        for batch in tqdm(loader):

            # clear gradient
            optimizer.zero_grad() 

            # clear loss
            loss = 0

            # run through summary
            for i in range(window, batch.shape[1] - (window + 1)):

                # context around tart word
                x = batch[:, [j for j in range(i - window, i + window + 1) if j != i]]

                # target word
                y = batch[:, i].flatten()

                # make prediction
                pred = model(x)

                # add loss
                loss += criterion(pred, y)

            # back propagate
            loss.backward()

            # improve parameters
            optimizer.step()



# document dimension reduction trainer
def doc_embed_trainer(model, loader, n_epochs, window, optimizer, criterion):

    # for every epoch
    for epoch in range(n_epochs):

        # run through all batches
        for batch in tqdm(loader):

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
    embed_size = 2 ** 8
    criterion = nn.CrossEntropyLoss()
    ds = WordDataset()
    model = WordModel(ds.tokenizer.vocab_size, embed_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 1
    window = 10
    loader = DataLoader(dataset=ds, batch_size=16, shuffle=True) 
    word_embed_trainer(model, loader, n_epochs, window, optimizer, criterion)
    torch.save(model.embed, f'../models/word_embed_{str(embed_size)}d.pt')
    
if __name__ == '__main__':
    main()
