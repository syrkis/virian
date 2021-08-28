# helpers.py
#   various auxiliary virian helpers and functions
# by: Noah Syrkis

# imports
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from tqdm import tqdm


# compute and save idf vector
def idf():
    with open('../../data/raw.tar', 'r') as f:
        D = TfidfVectorizer(f.readlines())


def tokenizer(line):
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    return bert_tokenizer(line)['input_ids']  


# call stack
def main():
    idf()

if __name__ == '__main__':
    main()
