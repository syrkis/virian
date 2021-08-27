# models.py
#   virian nlp models
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import WordDataset
from tqdm import tqdm


# word embedding model
class WordModel(nn.Module):

    # init call
    def __init__(self, vocab_size, dimensions):
    
        # init class parent
        super(WordModel, self).__init__()

        # layer widths
        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.fc2_counts = 100

        # declare embeddings layer
        self.embed = nn.Embedding(self.vocab_size, self.dimensions)

        # declare linear layer1
        self.fc1 = nn.Linear(self.dimensions, self.fc2_counts)

        # declare linear layer2
        self.fc2 = nn.Linear(self.fc2_counts, self.vocab_size)
 
    
    # forward pass function
    def forward(self, x):
        
        # word emebed x (window around target y)
        x = self.embed(x)
    
        # convert windoe to mean vector
        x = torch.mean(x, dim=1)

        # move through linear layer 1
        x = self.fc1(x)

        # move through linear layer 2
        x = self.fc2(x)

        # return word prediction
        return x


# document embedding model
class DocumentModel(nn.Module):

    # init call
    def __init__(self, vocab_size):
          
        # super class initialization
        super(DocumentModel, self).__init__()   

        # save vocabulary size for tf
        self.vocab_size = vocab_size
    
        # load precomputed 5d word emb.
        self.word_embed = torch.load('../models/word_embed_100d.pt')
        
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
