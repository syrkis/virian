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


# wikipedia class (tokenizes and scarpes, etc.)
class Wiki:

    date_format = variables['date_format']
    data_dir    = variables['data_dir']
    wiki_api    = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
    headers     = {"User-Agent": "nobr@itu.dk"}
    # tokenizer   = BPEmb(lang="multi", vs=10 ** 6, dim=300)
    
    def __init__(self, langs):
        self.langs      = langs
        self.start_date = "2015_07_01"
        self.end_date   = "2020_01_01"

    def get_dailies(self):
        for lang in self.langs:
            self._get_dailies_lang(lang)

    def get_texts(self):
        with Pool(len(self.langs)) as p:
            p.map(self._get_texts_lang, self.langs)

    def texts_to_toks(self):
        for lang in tqdm(self.langs):
            self._texts_to_toks_lang(lang)

    def _get_texts_lang(self, lang, fails = 0): # TODO: support cont.
        D = {"texts" : {}, "fails" : set()}
        wikipedia.set_lang(lang)
        titles = self._get_titles(lang)
        with tqdm(titles) as title_iter:
            for title in title_iter: # add fail found to tqdm
                if title not in D['fails']:
                    try:
                        D['texts'][title] = self.tokenizer.encode(wikipedia.page(title).summary)
                    except (wikipedia.exceptions.PageError, KeyError,
                            wikipedia.exceptions.DisambiguationError,
                            wikipedia.exceptions.WikipediaException,
                            json.decoder.JSONDecodeError) as e:
                        D['fails'].add(title)
                        title_iter.postfix(fail_ratio=len(D['fails']) / len(D['texts']))
                        continue
        with open(f"{self.data_dir}/wiki/text_{lang}.json", 'r') as f:
            json.dump(D, f, ensure_ascii=False)

    def _get_dailies_lang(self, lang): # TODO: support cont.
        D = {}
        for i in tqdm(range(self._str_to_delta(self.start_date, self.end_date))):
            date = self._add_days(self._to_date(self.start_date), i)
            url  = self._get_url(date, lang)
            res  = requests.get(url, headers=self.headers)
            day  = json.loads(res.text)['items'][0]['articles']
            D[self._to_str(date)] = [{k: v for k, v in entry.items() if k != 'rank'} for entry in day]
        with open(f"{self.data_dir}/wiki/days_{lang}.json", 'w') as f:
            f.write(json.dumps(D, ensure_ascii=False) + "\n")

    def _texts_to_toks_lang(self, lang): # one of migration func
        D = {"texts" : {}, "fails" : set()}
        hashes = [self._get_title_hash(title) for title in self._get_titles(lang)]
        with open(f"{self.data_dir}/wiki/text_{lang}.json", 'r') as f:
            texts = json.load(f)
        for _hash in tqdm(hashes):
            if _hash in texts:
                D[texts[_hash]['title']] = self.tokenizer.encode_ids(texts[_hash]['text'])
        D['fails'] = texts['__failed__']
        with open(f"{self.data_dir}/wiki/toks_{lang}.json", 'w') as f:
            json.dump(D, f)

    def _get_titles(self, lang):
        with open(f'{self.data_dir}/wiki/days_{lang}.json', 'r') as f:
            return set([text['article'] for day in f for text in json.loads(day)['data']]) # old schem

    def _titles_to_hash(self, titles):
        return {titles: titles_hash(title) for title in list(titles)}

    def _str_to_delta(self, start, end):
        return (self._to_date(end) - self._to_date(start)).days

    def _to_date(self, string):
        return datetime.strptime(string, self.date_format)

    def _add_days(self, date, days):
        return date + timedelta(days = days)

    def _to_str(self, date, sep='_'):
        return str(date).split()[0].replace('-', '/')

    def _get_url(self, date, lang):
        return f'{self.wiki_api}/{lang}.wikipedia.org/all-access/{self._to_str(date, "/")}'
        
    def _get_title_hash(self, title):
        return sha256((title).encode('utf-8')).hexdigest()


# dev calls
def main():
    pass

if __name__ == "__main__":
    main()

