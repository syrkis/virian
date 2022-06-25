# wiki.py
#   wiki daily top 1000 articles
# by: Noah Syrkis

# imports
from src.utils import variables
from datetime import datetime, timedelta
from bpemb import BPEmb
from multiprocessing import Pool
from hashlib import sha256
import os, json, requests, wikipedia
from tqdm import tqdm
import gensim
import fasttext
import numpy as np
from nltk.corpus import stopwords
import stopwordsiso as stopwords


# wikipedia class (tokenizes and scarpes, etc.)
class Wiki:

    date_format = variables['date_format']
    data_dir    = variables['data_dir']
    wiki_api    = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
    headers     = {"User-Agent": "nobr@itu.dk"}
    
    def __init__(self, conf):
        self.conf       = conf
        self.langs      = conf['langs']['train']
        self.start_date = "2015_07_01"
        self.end_date   = "2020_01_01"

    def get_dailies(self):
        for lang in self.langs:
            self.get_dailies_lang(lang)

    def get_texts(self):
        with Pool(len(self.langs)) as p:
            p.map(self.get_texts_lang, self.langs)

    def texts_to_toks(self, vocab_size): # bpemb (saves tokens)
        tokenizer = BPEmb(lang="multi", vs=vocab_size, dim=300)
        for lang in tqdm(self.langs):
            self._texts_to_toks_lang(lang, tokenizer, vocab_size)

    def text_to_vec(self): # fasttext and gensim converts article to mean vector embed rep
        with Pool(4) as p:
            p.map(self.text_to_vec_lang, self.langs)

    def text_to_vec_lang(self, lang):
        lang_stop_words = stopwords.stopwords(lang)
        vec_file = f"data/embs/wiki.{lang}.vec"
        embed = gensim.models.KeyedVectors.load_word2vec_format(vec_file)
        D = {"texts" : {}, "fails" : set()}
        hashes = [self._get_title_hash(title) for title in self._get_titles(lang)]
        with open(f"{self.data_dir}/wiki/text_{lang}.json", 'r') as f:
            texts = json.load(f)
        for _hash in tqdm(hashes):
            if _hash in texts:
                text = texts[_hash]['text']
                toks = fasttext.tokenize(text)
                embs = np.zeros(300)
                for tok in toks:
                    if tok in embed and tok not in lang_stop_words:
                        embs += embed[tok] # add all embeddings
                embs = embs / np.max(embs) if np.sum(np.abs(embs)) > 0 else embs
                D['texts'][texts[_hash]['title']] = embs.tolist() # mean embeddings
        D['fails'] = texts['__failed__']
        with open(f"{self.data_dir}/wiki/embs_{lang}.json", 'w') as f:
            json.dump(D, f)

    def _texts_to_toks_lang(self, lang, tokenizer, vocab_size): # one of migration function
        D = {"texts" : {}, "fails" : set()}
        hashes = [self._get_title_hash(title) for title in self._get_titles(lang)]
        with open(f"{self.data_dir}/wiki/text_{lang}.json", 'r') as f:
            texts = json.load(f)
        for _hash in tqdm(hashes):
            if _hash in texts:
                D[texts[_hash]['title']] = tokenizer.encode_ids(texts[_hash]['text'])
        D['fails'] = texts['__failed__']
        with open(f"{self.data_dir}/wiki/toks_{lang}_{vocab_size}.json", 'w') as f:
            json.dump(D, f)

    def get_texts_lang(self, lang, fails = 0): # TODO: support cont.
        D = {"__failed__" : []}
        wikipedia.set_lang(lang)
        titles = list(self._get_titles(lang))
        for i in tqdm(range(len(titles))): # add fail found to tqdm
            title = titles[i]
            if title not in D['__failed__']:
                try:
                    D[title] = wikipedia.page(title).summary
                except (wikipedia.exceptions.PageError, KeyError,
                        wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.WikipediaException,
                        json.decoder.JSONDecodeError) as e:
                    D['__failed__'].append(title)
                    continue
            if i % 1000 == 0:
                with open(f"{self.data_dir}/wiki/text_{lang}.json", 'w') as f:
                    json.dump(D, f, ensure_ascii=False)
        with open(f"{self.data_dir}/wiki/text_{lang}.json", 'w') as f:
            json.dump(D, f, ensure_ascii=False)

    def get_dailies_lang(self, lang): # TODO: support cont.
        D = {}
        for i in tqdm(range(self._str_to_delta(self.start_date, self.end_date))):
            date = self._add_days(self._to_date(self.start_date), i)
            url  = self._get_url(date, lang)
            res  = requests.get(url, headers=self.headers)
            day  = json.loads(res.text)['items'][0]['articles']
            D[self._to_str(date)] = [{k: v for k, v in entry.items() if k != 'rank'} for entry in day]
        with open(f"{self.data_dir}/wiki/days_{lang}.json", 'w') as f:
            f.write(json.dumps(D, ensure_ascii=False) + "\n")

    def _get_titles(self, lang):
        with open(f'{self.data_dir}/wiki/days_{lang}.json', 'r') as f:
            titles = set([entry['article'] for day in json.load(f).values() for entry in day])
        return titles

    def _titles_to_hash(self, titles):
        return {titles: titles_hash(title) for title in list(titles)}

    def _str_to_delta(self, start, end):
        return (self._to_date(end) - self._to_date(start)).days

    def _to_date(self, string):
        return datetime.strptime(string, self.date_format)

    def _add_days(self, date, days):
        return date + timedelta(days = days)

    def _to_str(self, date, sep='_'):
        return str(date).split()[0].replace('-', sep)

    def _get_url(self, date, lang):
        return f'{self.wiki_api}/{lang}.wikipedia.org/all-access/{self._to_str(date, "/")}'
        
    def _get_title_hash(self, title):
        return sha256((title).encode('utf-8')).hexdigest()


# dev calls
def main():
    pass

if __name__ == "__main__":
    main()

