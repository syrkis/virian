# datasets.py
#   virian web monitoring
# by: Noah Syrkis

# imports
import torch
import requests
from wikiapi import WikiApi
from transformers import BertTokenizer
from tqdm import tqdm


# 5d embeddings class
class EmbedDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        url = 'https://nostromo.syrkis.com'
        res = requests.get(url)
        self.text = res.text.split('***')[2]
    

         

# wikipedia daily top 1000 reads dataset
class WikiDataset(torch.utils.data.Dataset):

    wikiapi = WikiApi()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    
    def __init__(self, date):
        self.date = date
        headers = {"User-Agent": "virian@syrkis.com"}
        wiki = "en.wikipedia.org" 
        api = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{wiki}/all-access"
        url = f"{api}/{date}"  
        res = requests.get(url, headers=headers)
        self.data = res.json()['items'][0]['articles'] 
        for i in tqdm(range(4, len(self.data))):
            content = self.wikiapi.get_article(self.data[i]['article']).content
            summary = self.wikiapi.get_article(self.data[i]['article']).summary
            text = content + summary
            text = self.tokenizer.tokenize(text)
            text = text[: max(2 ** 9, len(text))]
            print(text) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# dev calls
def main():
    # dataset = WikiDataset("2020/08/01")
    dataset = EmbedDataset()

if __name__ == '__main__':
    main()
