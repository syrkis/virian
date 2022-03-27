# wiki.py
#   scarpes wiki project daily top 1000 api
# by: Noah Syrkis

# imports
from datetime import datetime, timedelta
import os
import json
import requests
import time
from tqdm import tqdm
from hashlib import sha256
import wikipedia


# run time function
def get_dailies(lang):
    file = f"../data/wiki/days/{lang}.json"
    api = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
    dones = get_dones(file)
    start_date = get_date(dones[-1].replace('_', '/') if dones else "2015/07/01")
    end_date = datetime.now() - timedelta(days=3) 
    date = start_date
    for _ in tqdm(range((end_date - start_date).days)):
        if get_str(date).replace('/', '_') in dones:
            date += timedelta(days = 1)
            continue
        date_str = get_str(date)
        headers = {"User-Agent": "nobr@itu.dk"}
        url = get_url(api, date_str, lang)
        res = requests.get(url, headers=headers).text
        data = json.loads(res)['items'][0]['articles']
        for sample in data:
            del sample['rank']
        D = {'date': get_str(date).replace('/', '_'), 'data': data}
        with open(file, 'a+') as f:
            json.dump(D, f)
            f.write('\n')
        date += timedelta(days = 1)


# get articles from dailies
def get_articles(lang):
    wikipedia.set_lang(lang)
    dailies_dir = f"../data/wiki/days"
    articles_dir = f"../data/wiki/text"
    with open(f"{articles_dir}/{lang}.json", "r") as f:
        corpus = json.loads(f.read())
    dailies = [file for file in os.listdir(dailies_dir) if file[-5:] == '.json']
    with open(f'{articles_dir}/{lang}_failed.txt', 'r') as f:
        failed = f.read().split()
    target_articles = set()
    with open(f"{dailies_dir}/{lang}.json", 'r') as f:
        for line in f.readlines():
            for article in json.loads(line)['data']:
                title = article['article']
                article_id = sha256((title).encode('utf-8')).hexdigest()
                if article_id not in corpus and title not in failed:
                    target_articles.add(article['article'])
    for idx, title in enumerate(tqdm(list(target_articles))):
        if idx % 100 == 0:
            with open(f"{articles_dir}/{lang}.json", "w") as f:
                json.dump(corpus, f)
        try:
            text = wikipedia.page(title).summary
            article_id = sha256((title).encode('utf-8')).hexdigest()
            corpus[article_id] = {"title": title, "text": text}
        except wikipedia.exceptions.PageError:
            open(f"{articles_dir}/{lang}_failed.txt", 'a').write(f"{title}\n")
            pass
        except KeyError:
            open(f"{articles_dir}/{lang}_failed.txt", 'a').write(f"{title}\n")
            pass
        except wikipedia.exceptions.DisambiguationError:
            open(f"{articles_dir}/{lang}_failed.txt", 'a').write(f"{title}\n")
            pass
        except json.decoder.JSONDecodeError:
            open(f"{articles_dir}/{lang}_failed.txt", 'a').write(f"{title}\n")
            pass
        except wikipedia.exceptions.WikipediaException:
            open(f"{articles_dir}/{lang}_failed.txt", 'a').write(f"{title}\n")
            pass
    with open(f"{articles_dir}/{lang}.json", "w") as f:
        json.dump(corpus, f)


# helpers
get_url = lambda api, date, lang: f'{api}/{lang}.wikipedia.org/all-access/{date}'
get_date = lambda s: datetime.strptime(s, "%Y/%m/%d")
get_str = lambda d: str(d).split()[0].replace('-', '/')

def  get_dones(file):
    exist = os.path.exists(file)
    if exist:
        with open(file, 'r') as f:
            dones = [json.loads(sample)['date'] for sample in f.readlines()]
        return sorted(dones)
    return None


# dev calls
def main():
    pass

if __name__ == "__main__":
    main()

