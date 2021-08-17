# models.py
#   virian nlp models
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from tqdm import tqdm


# word embedding model
class WordModel(nn.Module):
    def __init__(self, vocab_size, dimensions):
        super(WordModel, self).__init__()
        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.embed = nn.Embedding(
            self.vocab_size,
            self.dimensions
        )
        self.fc = nn.Linear(
            self.dimensions,
            self.vocab_size
        )  
        
    def forward(self, x):
        x = self.embed(x)
        x = torch.sum(x, dim=1)
        x = self.fc(x)
        return x


# document embedding model
class DocumentModel(nn.Module):
    def __init__(self, vocab_size):
        super(DocumentModel, self).__init__()   
        self.vocab_size = vocab_size
        self.embed = torch.load('../models/word_embed_5d.pt')
        self.tfidf = torch.load('../models/idf.pt') # tfidf for weighted sum of embeddings

    def forward(self, x):
        tfidf = self.tf(x) * self.tfidf
        n = torch.sum(tfidf, dim=1)[:, None]
        tfidf /= n
        print(tfidf)

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
    model = DocumentModel(30522)
    ds = Dataset()
    loader = DataLoader(dataset=ds, batch_size=32)
    # idf = torch.ones(30522)
    for batch in tqdm(loader):
        model.forward(batch)
        # idf = (len(loader.dataset) + len(idf)) * (1 / model.idf(batch, idf))
    # torch.save(torch.log(idf), '../models/idf.pt')

if __name__ == '__main__':
    main()
