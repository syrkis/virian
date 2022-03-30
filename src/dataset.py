# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports from src import 
from src.utils import hypers, load, title_hash, tokenize
from transformers import AutoTokenizer
from src.ess import construct_factors
import torch
from tqdm import tqdm
from itertools import cycle


# wiki dataset
class Dataset(torch.utils.data.IterableDataset):

    vocab_size  = hypers['vocab_size']  # TODO: multilingual vocab?
    sample_size = hypers['sample_size'] # 128 word wiki summaries
    tokenizer   = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __init__(self):
        self.text      = load('text')  # wiki summaries
        self.days      = load('days')  # wiki dailies
        # self.ess       = construct_factors() # ess factors
        
    def process_data(self):
        for lang, days in self.days.items():
            for day in days:
                yield self.construct_sample(day['date'], lang, day['data'])

    def construct_sample(self, date, lang, data):
        X = torch.zeros((1000, self.sample_size))
        W = torch.zeros(1000)
        # Y = self.ess[date]

        for idx, article in enumerate(data):
            title = title_hash(article['article'])
            if title in self.text[lang]:
                text = self.text[lang][title]['text']
                toks = tokenize(text, self.tokenizer)
                X[idx] += toks
                #W [idx] += article['views']

        return X #  W # , Y

    def get_stream(self):
        return cycle(self.process_data())

    def __iter__(self):
        return self.get_stream()


# dev stack
def main():
    dataset = Dataset()

if __name__ == "__main__":
    main()

