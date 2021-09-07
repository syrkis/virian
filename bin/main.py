# main.py
#   describes articles relative to haidt dimensions  
# by: Noah Syrkis

# imports
import torch
from dataset import Dataset
from model import Model


# describe articles on haidth dimensions
def describe(model, articles):
    for article in articles:
        _, comp = model.foward(article)


# call stack
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = Dataset(device, "2020/10/10", 2 ** 13)
    model = Model(ds.model.distilbert.embeddings.word_embeddings.weight.shape[1]).to(device)
    model.load_state_dict(torch.load('../models/model.pt'))
    model.eval()

if __name__ == '__main__':
    main()
