# models.py
#   virian nlp models
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# document embedding model
class Model(nn.Module):

    # init call
    def __init__(self, embed_dim):
          
        # super class initialization
        super(DocumentModel, self).__init__()   

        # initialize doc embeddings (using Linear due to word embed)
        self.fc1 = nn.Linear(100, 5)

        # decompression layer
        self.fc2 = nn.Linear(5, 100)
    
        # load in precomputed idf vector
        self.tfidf = torch.load('../models/idf.pt')

    # forward pass 
    def forward(self, x):

        # compress x
        h = self.fc1(x) 

        # decompress x
        x = self.fc2(h)
        
        # debug prints
        return x, h
        

# dev calls
def main():
    from torch.utils.data import DataLoader
    from dataset import Dataset
    from tqdm import tqdm
    vocab_size = 30522
    model = DocumentModel(vocab_size)
    ds = Dataset()
    loader = DataLoader(dataset=ds, batch_size=32)
    # idf = torch.ones(vocab_size)
    for batch in tqdm(loader):
        model(batch)
        # idf = model.idf(batch, idf)
    # idf = torch.log((len(loader.dataset) + vocab_size) * (1 / idf))
    # torch.save(torch.log(idf), '../models/idf.pt')

if __name__ == '__main__':
    main()
