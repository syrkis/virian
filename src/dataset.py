# datasets.py
#   virian wiki summary datasets
# by: Noah Syrkis

# imports
from datasets import load_dataset

# main
def main():
    data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
    ds = load_dataset(
                "json", data_files=data_files, split="train", streaming=True
                )
    for i in range(10):
        print(next(iter(ds)))

if __name__ == '__main__':
    main()
