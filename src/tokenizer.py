# vim tokenizer.py
#   virian tokenizer
# by: Noah Syrkis

# imports
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers


# train tokenizer
def train_tokenizer():
    from dataset import Dataset
    ds = Dataset()

def get_tokenizer():
    pass

tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.UnigramTrainer(
    vocab_size=1000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"]
)

data = [
    "saliva is the most versatile fluid of the human body",
    "on latitudes this low the sun sets orthgonally to the horizon",
    "on latitudes this low the sun sets orthgonally to the horizon",
    "on latitudes this low the sun sets orthgonally to the horizon",
    "on latitudes this low the sun sets orthgonally to the horizon",
    "on latitudes this low the sun sets orthgonally to the horizon",
    "suns do not bring the likes of I",
    "i reamin hhere",
    "I thoughtthe heavs would fall upon my head"
]

tokenizer.train_from_iterator(data, trainer=trainer)

print(tokenizer.get_vocab())
