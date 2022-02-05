# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from torch.utils.data import IterableDataset
from itertools import cycle
from src.helpers import get_s3


# dataset
class Dataset(IterableDataset):

    vocab_size = 2 ** 12
    s3 = get_s3()
    remote = s3.get_object(Bucket="data", Key="20200301.en.100k")

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


def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

