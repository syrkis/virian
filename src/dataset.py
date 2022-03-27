# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
from src import utils
import torch
from tqdm import tqdm
from itertools import cycle


# wiki dataset
class Dataset(torch.utils.data.IterableDataset):

    vocab_size  = utils.hypers['vocab_size']  # TODO: multilingual vocab
    sample_size = utils.hypers['sample_size'] # 128 word wiki summaries

    def __init__(self, tokenizer=None):
        self.texts     = utils.load('texts') # wiki summaries
        self.days      = utils.load('days')  # wiki dailies
        self.ess       = utils.load('ess')   # ess factors
        self.tokenizer = tokenizer
        
    def process_data(self):
        if self.tokenizer:
            for lang, data in self.days.items():
                for date, text in data.items():
                    yield self.construct_sample(date, lang, text)
        else:
            for title, text in self.texts.items():
                yield text

    def construct_sample(self, date, lang, text):
        X = torch.zeros((1000, self.sample_size))
        W = torch.zeros(1000)
        Y = self.ess[date]

        for idx, text in enumerate(articles):
            if utils.title_hash(text['title']) in self.articles[lang]:
                W[idx] += article['views']
                X[idx] += self.tokenizer.vectorize(text['text'])

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

