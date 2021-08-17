# datasets.py
#   virian web monitoring
# by: Noah Syrkis

# imports
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import linecache


# wiki summary dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.n_samples = 2 ** 10 # 5_315_384
        self.n_words = 2 ** 7

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        line = linecache.getline('../../data/raw.tar', idx)
        tokens = self.tokenizer(line)['input_ids']
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
        tmp = batch.shape
    
    

if __name__ == '__main__':
    main()
