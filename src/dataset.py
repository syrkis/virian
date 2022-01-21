# dataset.py
#   virian dataset script
# by: Noah Syrkis

# by: Noah Syrkis
import numpy
import torch
import hub
import tensorflow_datasets as tfds


# dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, lang):
        self.lang = lang
        self.data = hub.load(f"hub://syrkis/wiki.{lang}") 

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
    
    def __repr__(self):
        return self.lang


def main():
    ds = Dataset('da')

if __name__ == "__main__":
    main()
