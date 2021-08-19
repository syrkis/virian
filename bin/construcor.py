# construcutr.py
#   makes wiki summary word indexes
# by: Noah Syrkis

# imports
import torch
from tokenizer import tokenizer
from tqdm import tqdm


# loop through data constructing vocabulay
def constructor():
    n_samples = 5_315_384
    vocab = set()
    tmp = set()
    with open('../../data/raw.tar', 'r') as f:
        for i in range(n_samples):
            summary = tokenizer(f.readline())
            for word in summary:
                if word in tmp and word not in vocab:
                    vocab.add(word) 
                if word.isalpha() and word not in tmp:
                    tmp.add(word)
            if i % 1000 == 0:
                print(len(vocab))
    vocab = sorted(list(vocab))
    print(len(vocab))


# call stack
def main():
    constructor()

if __name__ == '__main__':
    main()
