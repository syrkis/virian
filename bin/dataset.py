# datasets.py
#   virian web monitoring
# by: Noah Syrkis

# imports
import torch
from tqdm import tqdm
import linecache
from tokenizer import tokenizer


# wiki summary dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.n_samples = 2 ** 15 # 5_315_384
        self.n_words = 2 ** 7

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        line = linecache.getline('../../data/raw.tar', idx)
        tokens = tokenizer(line)
        tokens = tokens[: min(self.n_words, len(tokens))]
        if len(tokens) < self.n_words:
            tmp = [0 for _ in range(self.n_words - len(tokens))] 
            tmp.extend(tokens)
            tokens = tmp
        return torch.tensor(tokens)


# dev calls
def main():
    ds = Dataset()
    loader = torch.utils.data.DataLoader(dataset=ds, batch_size=32, shuffle=True)
    for batch in tqdm(loader):
        print(batch)
    
    

if __name__ == '__main__':
    main()
