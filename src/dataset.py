# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
import os, re, torch, fileinput, pickle, json
from boto3.session import Session
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
from itertools import cycle


# dataset
class Dataset(torch.utils.data.IterableDataset):
    """
    This data set is for training the virian topic model.
    For now it either loads a vocab from s3 or computes are uploads one.
    Arguably tokenization should be done with huggingface.
    This setup is however simpler to get started with.
    """

    unk = "<UNK>"
    vocab_size = 2 ** 12
    vocab_files = [f for f in "vocab word_to_idx idx_to_word".split()]

    def __init__(self, vocab_trained=True):
        self.vocab_trained = vocab_trained
        self.s3 = self.get_s3()
        path = '../data/joseph_conrad'
        self.files = tuple([f"{path}/{f}" for f in os.listdir(path) if f[-3:] == 'txt'])
        self.get_vocab()
        
    def parse_file(self, files):
        with fileinput.input(files=(files)) as f:
            for line in f:
                yield self.bag_of_words(line) if self.vocab_trained else line

    def get_stream(self, files):
        return cycle(self.parse_file(files))

    def train_vocab(self):
        freqs = Counter()
        for sample in self.parse_file(self.files):
            freqs.update(self.tokenize(sample))
        vocab = [w[0] for w in freqs.most_common(self.vocab_size - 1)] # -1 for unk
        vocab += [self.unk] # add unk
        word_to_idx = {w: idx for idx, w in enumerate(vocab)}
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        self.vocab_trained = True
        self.push_vocab(vocab, word_to_idx, idx_to_word)
        self.vocab, self.word_to_idx, self.idx_to_word = vocab, word_to_idx, idx_to_word

    def tokenize(self, text): # TODO: switch to hugging face?
        return re.sub(r'[^a-zA-Z ]', ' ', text).lower().split()

    def bag_of_words(self, line):
        vec = torch.zeros(self.vocab_size)
        tokens = self.tokenize(line)
        for tok in tokens:
            tok = tok if tok in self.vocab else self.unk # insert unks
            vec[self.word_to_idx[tok]] += 1
        return vec

    def get_vocab(self):
        if self.vocab_trained:
            for file in self.vocab_files:
                obj = self.s3.get_object(Bucket='prepro', Key=file)["Body"].read()
                data = pickle.loads(obj)
                exec(f"self.{file} = data")
        else:
            self.train_vocab()


    def push_vocab(self, *args): # TODO: tokenizer versioning?
        for idx, obj in enumerate(args):
            self.s3.put_object(Bucket="prepro",
                    Body=pickle.dumps(obj),
                    Key=self.vocab_files[idx])

    def get_s3(self):
        session = Session()
        client = session.client('s3',
                region_name='AMS3',
                endpoint_url='https://virian.ams3.digitaloceanspaces.com',
                aws_access_key_id='QA3DDQQ6ITF3JMXZOK3H',
                aws_secret_access_key='6Kjt6zx38aOBlOf2HUnxcq9zeA30iVY1Zqs3X3XP03g')
        return client

    def __iter__(self):
        return self.get_stream(self.files)


def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()

