# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# import
from src import WikiDataset, ESSDataset, TopicModel, ValueModel, Tokenizer, train
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# call stack
def main():
    tokenizer = Tokenizer(trained=True)
    ds = WikiDataset(tokenizer)
    loader = DataLoader(dataset=ds, batch_size=30)
    topic_model = TopicModel(ds.vocab_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(topic_model.parameters())
    train(loader, topic_model, optimizer, criterion)

if __name__ == "__main__":
    main()
