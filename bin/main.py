from csv import writer
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', model_max_len=512)
data = open('data/raw.tar', 'r').readlines()
batch_size = 2 ** 14

with open('data/tok.csv', 'a+', newline="") as f:
    writer = writer(f)
    for i in tqdm(range(0, (len(data) // batch_size) * batch_size, batch_size)):
        batch = tokenizer.batch_encode_plus(data[i: i + batch_size], truncation=True)['input_ids']
        writer.writerows(batch)
        

