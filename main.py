# main.py
#   describes articles relative to haidt dimensions  
# by: Noah Syrkis

# imports
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model
from datetime import datetime, timedelta


# describe articles on haidth dimensions
def describe(model, loader):
    for batch in loader:
        _, comp = model(batch)
        print(comp)
        break


# call stack
def main():
    haidt_dims = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    ds = Dataset(local=False)
    loader = DataLoader(dataset=ds, batch_size=2 ** 2)
    model = Model(ds.model.distilbert.embeddings.word_embeddings.weight.shape[1])
    model.load_state_dict(torch.load('../models/model.pt'))
    model.eval()
    describe(model, loader)

if __name__ == '__main__':
    main()
