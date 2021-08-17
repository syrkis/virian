# tfidf.py
#   tfidf for wiki summaries
# by: Noah Syrkis

# imports
import torch
from torch.utils.data import DataLoader
from dataset import Dataset


# calcualte tfidf
def tf(loader):
    
    for batch in loader:
        print(batch.shape)
    

def main():
    ds = Dataset()
    loader = DataLoader(dataset=ds, batch_size=32, shuffle=True)
    tf(loader)

if __name__ == '__main__':
    main()
