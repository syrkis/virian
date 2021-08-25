# models.py
#   virian nlp models
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Dataset
from tqdm import tqdm


# word embedding model
class WordModel(nn.Module):
    """
    Word embedding trainer class as
    auxiliary from language model.
    For now word embeddings are 100
    dimensional.
    """

    # init call
    def __init__(self, vocab_size, dimensions):
    
        # init class parent
        super(WordModel, self).__init__()

        # layer widths
        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.fc2_counts = 100

        # declare embeddings layer
        self.embed = nn.Embedding(
            self.vocab_size,
            self.dimensions
        )

        # declare linear layer1
        self.fc1 = nn.Linear(
            self.dimensions,
            self.fc2_counts
        )

        # declare linear layer2
        self.fc2 = nn.Linear(
            self.fc2_counts,
            self.vocab_size
        )
 
    
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
    """
    Document embedding constructor class.
    currently loads in 5d word embeddings
    and wiki summary corpus idf, with
    intention of performing tf-idf
    weigthed sum for word embeddings.
    Considering switching word embeddings
    to 100d and decreasing with auto encoder.
    """

    # init call
    def __init__(self, vocab_size):
          
        # super class initialization
        super(DocumentModel, self).__init__()   

        # save vocabulary size for tf
        self.vocab_size = vocab_size
    
        # load precomputed 5d word emb.
        self.word_embed = torch.load('../models/word_embed_100d.pt')
        
        # initialize doc embeddings (using Linear due to word embed)
        self.fc1 = nn.Linear(
            100,
            5
        )

        self.fc2 = nn.Linear(
            5,
            100
        ) 
    
        # load in precomputed idf vector
        self.tfidf = torch.load('../models/idf.pt')

    # forward pass 
    def forward(self, x):

        # compute tf-idf score for samples in batch
        tfidf = self.tf(x) * self.tfidf

        # normalize tfidf vector (perhaps skip)
        tfidf /= torch.sum(tfidf, dim=1)[:, None]

        # tfidf weight matrix
        w = torch.zeros(x.shape) 

        # loop through batches
        for i in range(x.shape[0]):

            # weights for each word emb.
            w[i] += tfidf[i][x[i]]
             
        # get word embeddings
        x = self.word_embed(x) 

        # weighted mean of embeddings (broadcast)
        x *= w[:, :, None]
    
        # sum pre weighted embeddings
        x = torch.sum(x, dim=1)

        # compress x
        x = self.fc1(x) 

        # decompress x
        x = self.fc2(x)
        
        # debug prints
        return x
        

    def tf(self, x):
        o = torch.zeros((x.shape[0], x.shape[1], self.vocab_size)) 
        o.scatter_(2, x.unsqueeze(2), 1)
        tf = torch.sum(o, dim=1)  
        return tf

    def idf(self, x, idf):
        tf = self.tf(x) 
        idf += torch.sum((tf != 0).int(), dim=0)
        return idf
 

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
