# articles.py
#   scrapes wiki articles
# by: Noah Syrkis

# imports
import requests as req
import json, os
from hashlib import sha256
import wikipedia
from tqdm import tqdm


# wiki article scraper
def get_articles(lang):
    wikipedia.set_lang(lang)
    dailies_dir = f"../data/dailies/{lang}"
    articles_dir = f"../data/articles"
    with open(f"{articles_dir}/{lang}.json", "r") as f:
        corpus = json.loads(f.read())
    dailies = [file for file in os.listdir(dailies_dir) if file[-5:] == '.json']
    with open(f'{articles_dir}/{lang}_failed.txt', 'r') as f:
        failed = f.read().split()
    target_articles = set()
    for daily in tqdm(dailies):
        with open(f"{dailies_dir}/{daily}", 'r') as f:
            for article in json.load(f):
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
        except:
            print(lang)
    with open(f"{articles_dir}/{lang}.json", "w") as f:
        json.dump(corpus, f)




# dev calls
def main():
    dailies("2020/11/02", "de")

if __name__ == "__main__":
    main()
