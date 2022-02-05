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
    vocab_files = [f for f in "vocab word_to_idx idx_to_word".split()]

    def __init__(self, vocab_trained=True):
        self.vocab_trained = vocab_trained
        self.s3 = get_s3()
        self.data_source = self.s3.get_object(Bucket="data", Key="20200301.en.100k")
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.get_vocab()
        
    def get_stream(self, remote_data):
        return cycle(self.parse_stream(remote_data))

    def parse_stream(self, remote_data):
        body = remote_data["Body"]
        for article in body:
            yield self.bag_of_words(article) if self.vocab_trained else article

    def tokenize(self, text): # TODO: switch to hugging face?
        return re.sub(r'[^a-zA-Z ]', ' ', text).lower().split()

    def bag_of_words(self, line):
        tokens = self.tokenize(line.decode(errors='replace'))
        vec = torch.zeros(self.vocab_size)
        for tok in tokens:
            tok = tok if tok in self.vocab else self.unk # insert unks
            vec[self.word_to_idx[tok]] += 1
        return vec

    def get_vocab(self):
        if self.vocab_trained:
            for file in self.vocab_files:
                obj = self.s3.get_object(Bucket='prepro', Key=file)["Body"].read()
                data = pickle.loads(obj)
                exec(f"self.{file} = data") # declare vocab and dicts in self
        else:
            self.train_vocab()
 
    def train_vocab(self):
        freqs = Counter()
        for sample in self.parse_files(self.files):
            freqs.update(self.tokenize(sample))
        vocab = [w[0] for w in freqs.most_common(self.vocab_size - 1)] + [self.unk]
        word_to_idx = {w: idx for idx, w in enumerate(vocab)}
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        self.vocab_trained = True
        self.push_vocab(vocab, word_to_idx, idx_to_word)
        self.vocab, self.word_to_idx, self.idx_to_word = vocab, word_to_idx, idx_to_word

    def push_vocab(self, *args): # TODO: tokenizer versioning?
        for idx, obj in enumerate(args):
            self.s3.put_object(Bucket="prepro",
                    Body=pickle.dumps(obj),
                    Key=self.vocab_files[idx])

    def __iter__(self):
        return self.get_stream(self.data_source)


def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

