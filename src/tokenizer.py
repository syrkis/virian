# tokenizer.py
#   virian tokenizer and vocabulary getter / trainer
# by: Noah Syrkis

# imports
from src.dataset import Dataset
from src.helpers import get_s3
from collections import Counter
from itertools import islice
from tqdm import tqdm
import pickle, re, torch


# tokenizer class
class Tokenizer(Dataset):

    unk = "<UNK>"

    def __init__(self, trained=True):
        super().__init__()
        self.trained = trained
        self.word_to_idx, self.idx_to_word = self.get_vocab()

    def get_vocab(self):
        if self.trained:
            download = lambda key: pickle.loads(self.s3.get_object(Bucket='prepro', Key=key)["Body"].read())
            return list(map(download, ["word_to_idx", "idx_to_word"]))
        return self.train_vocab()
 
    def train_vocab(self):
        freqs = Counter()
        n = len(self.article_files)
        with tqdm(islice(self.process_data(self.article_files), n), unit="langs", total=10000) as lang:
            for sample in lang:
                freqs.update(self.tokenize(sample))
        vocab = [w[0] for w in freqs.most_common(self.vocab_size - 1)] + [self.unk]
        word_to_idx = {w: idx for idx, w in enumerate(vocab)}
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        self.push_vocab(word_to_idx, idx_to_word)
        return word_to_idx, idx_to_word

    def push_vocab(self, word_to_idx, idx_to_word): # TODO: tokenizer versioning?
        upload = lambda k, v: self.s3.put_object(Bucket="prepro", Body=pickle.dumps(v), Key=k)
        map(upload, [("word_to_idx", word_to_idx), ("idx_to_word", idx_to_word)])

    def tokenize(self, text): # TODO: switch to hugging face? deffinately make better regex
        return re.sub(r'[^a-zA-Z ]', ' ', text).lower().split()

    def bag_of_words(self, sample): # TODO: ehm, bag of words is probably too stupid
        tokens = self.tokenize(sample)
        vec = torch.zeros(self.vocab_size)
        for tok in tokens:
            tok = tok if tok in self.word_to_idx else self.unk # insert unks
            vec[self.word_to_idx[tok]] += 1
        return vec


# dev calls
def main():
    pass

if __name__ == "__main__":
    main()
