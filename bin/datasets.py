# datasets.py
#   virian wiki summary datasets
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import linecache
from transformers import AutoTokenizer



# wiki summary dataset
class WordDataset(Dataset):


    # run on class instanciation
    def __init__(self):

        # load in BERT tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        # dictate size of data set 
        self.n_samples = 10 ** 5 # 5_315_384

        # number of words in summary (truncate / pad)
        self.n_words = 2 ** 7


    # defining the concept of dataset length
    def __len__(self):

        # return sample count as length
        return self.n_samples


    # def fine what dataset[idx] returns
    def __getitem__(self, idx):

        # get tokenized and truncated / padded summary
        tokens = self._tokens(idx)

        # return sample as tensor
        return tokens


    # hepler function to tokenize and truncate / pad
    def _tokens(self, idx):

        # load line and prep data file for quick access
        line = linecache.getline('../../data/raw.tar', idx)

        # tokenize summary
        tokens = self.tokenizer(line)['input_ids']

        # truncate if too long
        tokens = tokens[: min(self.n_words, len(tokens))]

        # if summary is to short
        if len(tokens) < self.n_words:

            # pad with this many 0's
            tmp = [0 for _ in range(self.n_words - len(tokens))] 

            # create padded array 
            tmp.extend(tokens)

            # assign tokens var to said array
            tokens = tmp

        # return tokens
        return torch.tensor(tokens)



# document dataset class
class DocumentDataset(WordDataset):


    # on initialization
    def __init__(self):
        
        # do same as for WordDataset
        super(DocumentDataset, self).__init__()

        # load in word embeddings
        self.embed = torch.load('../models/word_embed_100d.pt')

        # load in idf array for for weighted average doc embed.
        self.idf = torch.tensor(np.loadtxt('../models/idf.csv', delimiter='\n'))


    # define ds[idx] meaning 
    def __getitem__(self, idx):

        # get tokenized and truncated / padded summary 
        tokens = self._tokens(idx)
    
        # tf-idf
        tfidf = self._tf(tokens) * self.idf

        # embedding weights
        w = tfidf[tokens]

        # normalize weights
        w /= torch.sum(tokens)

        # embed samples
        embeds = self.embed(tokens) 

        # apply weightes to embeddings
        embeds *= w[:, None]

        # add embeddings
        embeds = torch.sum(embeds, dim=0)

        # return embeds
        return embeds 


    # term freq sample calculator     
    def _tf(self, tokens):
        
        # zeros matrix for one tf freq counts
        o = torch.zeros((tokens.shape[0], self.tokenizer.vocab_size))

        # populate o
        o.scatter_(1, tokens.unsqueeze(1), 1)
        
        # return tf vector
        return torch.sum(o, dim=0)


    # idf calcualtor
    def tfidf(self, batch, idf):

        # term freq of batch
        for i in range(batch.shape[0]):
        
            # calculate tf for sample
            tf = self._tf(batch[i])

            # add counts to idf
            idf += (tf != 0).int()

        # return new idf for next round
        return idf



# idf constructer
def idf(words_loader, docs_dataset, idf):
    for batch in tqdm(words_loader):
        idf = docs_dataset.tfidf(batch, idf)
    idf = torch.log((len(words_loader.dataset) + docs_dataset.tokenizer.vocab_size) / idf)
    idf = idf.numpy()
    idf.tofile('../models/idf.csv', sep='\n')


# dev calls
def main():
    # words_dataset = WordDataset()
    docs_dataset = DocumentDataset()
    # words_loader = DataLoader(dataset=words_dataset, batch_size=32, shuffle=True)
    docs_loader = DataLoader(dataset=docs_dataset, batch_size=32, shuffle=True)
    # idf(words_loader, docs_dataset, torch.ones(words_dataset.tokenizer.vocab_size))
    for batch in docs_loader:
        print(batch.shape)
        break

if __name__ == '__main__':
    main()
