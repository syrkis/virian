# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
import os, re, torch, pickle
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from itertools import cycle
from src.helpers import get_s3


# dataset
class Dataset(torch.utils.data.IterableDataset):

    unk = "<UNK>"
    vocab_size = 2 ** 12

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.s3 = get_s3()
        self.data_source = self.s3.get_object(Bucket="data", Key="20200301.en.100k")
        
    def get_stream(self, remote_data):
        return cycle(self.parse_stream(remote_data))

    def parse_stream(self, remote_data):
        for article in remote_data["Body"]: # s3 bucket
            yield self.tokenizer.bag_of_words(article) if self.tokenizer else article

    def __iter__(self):
        return self.get_stream(self.data_source)


def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

