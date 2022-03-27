# tokenizer.py
#   virian tokenizer and vocabulary getter / trainer
# by: Noah Syrkis

# imports
import json
from src import utils
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


# setup
def train_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SEP]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(utils.text_iter())
    return tokenizer


# dev calls
def main():
    pass

if __name__ == "__main__":
    main()
