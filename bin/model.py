# model.py
#   virian nlp models
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn


# document embeddings trainer
class Model(nn.Module):
        
    vocab_size = 30522
    dimensions = 5

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(
            self.vocab_size,
            self.dimensions
        )
        self.fc = nn.Linear(
            self.dimensions,
            self.vocab_size
        )  
        
    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x


# dev calls
def main():
    model = Model()

if __name__ == '__main__':
    main()
