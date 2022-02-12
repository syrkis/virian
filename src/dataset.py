# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from torch.utils.data import Dataset, IterableDataset
from itertools import cycle
from src.helpers import get_s3
import pandas as pd


# wiki dataset
class WikiDataset(IterableDataset):

    vocab_size = 2 ** 12
    s3 = get_s3()
    remote = s3.get_object(Bucket="data", Key="wiki/20200301.en.100k")

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        
    def get_stream(self, remote):
        return cycle(self.parse_stream(remote))

    def parse_stream(self, remote):
        for article in remote["Body"]: # s3 bucket
            if self.tokenizer:
                yield self.tokenizer.bag_of_words(article.decode(errors="replace"))
            else:
                yield article.decode(errors="replace")

    def __iter__(self):
        return self.get_stream(self.remote)


# ess dataset
class ESSDataset(Dataset):

    def __init__(self):
        self.data = pd.read_csv("../data/dump/ess/r_7_8_9_rel_sub/ESS1-9e01_1.csv", dtype="object")
        self.data["inwyye"] = self.data["inwyye"].astype(int)

    def __len__(self):
        return len(pd.unique(self.data["inwyye"]))

    def __getitem__(self, idx):
        return self.data.loc[self.data.inwyye == idx + min(self.data["inwyye"])]


# dev stack
def main():
    dataset = ESSDataset()
    
if __name__ == "__main__":
    main()

