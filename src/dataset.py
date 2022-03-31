# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports from src import 
from src.utils import hypers, load, title_hash, tokenize, paths
from transformers import AutoTokenizer
from src.ess import construct_factors
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle


# wiki dataset
class Dataset(torch.utils.data.IterableDataset):

    vocab_size  = hypers['vocab_size']  # TODO: multilingual vocab?
    sample_size = hypers['sample_size'] # 128 word wiki summaries
    tokenizer   = AutoTokenizer.from_pretrained(paths["tokenizer"])

    def __init__(self):
        self.text      = load('text')  # wiki summaries
        self.days      = load('days')  # wiki dailies
        # self.ess       = construct_factors() # ess factors
        
    def process_data(self):
        for lang, days in self.days.items():
            for day in days:
                yield self.construct_sample(day['date'], lang, day['data'])

    def construct_sample(self, date, lang, data):
        titles = [title_hash(article['article']) for article in data] 
        texts  = [self.text[lang][title]['text'] for title in titles]
        X      = tokenize(texts, self.tokenizer)
        views  = torch.tensor([article['views'] for article in data])
        W      = F.pad(views, pad=(0,1000-views.shape[0]))
        # Y = self.ess[date]
        return X#, W # , Y

    def get_stream(self):
        return cycle(self.process_data())

    def __iter__(self):
        return self.get_stream()


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

