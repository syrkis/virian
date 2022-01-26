# dataset.py
#   virian dataset class
# by: Noah Syrkis

# imports
import os, re, torch
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F


# dataset
class Dataset(torch.utils.data.Dataset):

    unk = "<UNK>"
    vocab_size = 2 ** 8 # 99 words plus unk

    def __init__(self):
        path = "../data/joseph_conrad"
        files = [f for f in os.listdir(path) if f[-3:] == "txt"]
        data = [open(f"{path}/{f}").read().split("***")[2] for f in files]
        self.data = list(map(self._tokenize, data))
        self.vocab, self.word_to_idx, self.idx_to_word = self._train_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = [word if word in self.vocab else self.unk for word in self.data[idx]]
        bow = torch.tensor(list(set([self.word_to_idx[word] for word in sample])))
        one_hot = torch.sum(F.one_hot(bow), dim=0).to(torch.float32)
        return one_hot

    def _tokenize(self, text):
        return re.sub(r'[^a-zA-Z ]', ' ', text).lower().split()

    def _train_vocab(self):
        freqs = Counter([w for s in self.data for w in s])
        vocab = sorted(freqs.items(), key=lambda x: x[1])[-(self.vocab_size-1):]
        vocab = [v[0] for v in vocab] + [self.unk]
        word_to_idx = {w: idx for idx, w in enumerate(vocab)}
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        return vocab, word_to_idx, idx_to_word


# call stack
def main():
    dataset = Dataset()
    
if __name__ == "__main__":
    main()
