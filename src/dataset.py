# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src import utils
from src.ess import construct_factors
import torch
from tqdm import tqdm
from itertools import cycle


# wiki dataset
class Dataset(torch.utils.data.IterableDataset):

    vocab_size  = utils.hypers['vocab_size']  # TODO: multilingual vocab?
    sample_size = utils.hypers['sample_size'] # 128 word wiki summaries

    def __init__(self, tokenizer):
        self.text      = utils.load('text')  # wiki summaries
        self.days      = utils.load('days')  # wiki dailies
        self.ess       = construct_factors() # ess factors
        self.tokenizer = tokenizer
        
    def process_data(self):
        if self.tokenizer:
            for lang, day in self.days.items():
                for entry in day.values()['data']:
                    yield self.construct_sample(day['date'], lang, entry['text'])

    def construct_sample(self, date, lang, text):
        return text
        X = torch.zeros((1000, self.sample_size))
        W = torch.zeros(1000) + article['views']
        Y = self.ess[date]
        for idx, text in enumerate(articles):
            if utils.title_hash(text['title']) in self.articles[lang]:
                X[idx] += self.tokenizer.encode(text['text'][:self.sample_size])
        return X, W, Y

    def get_stream(self):
        return cycle(self.process_data())

    def __iter__(self):
        return self.get_stream()


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

