# train.py
#   virian training
# by: Noah Syrkis

# imports
import torch


def train(loader):
    for batch in loader:
        print(batch)
        break


def main():
    import hub
    ds = hub.load("hub://syrkis/wiki.da")
    loader = ds.pytorch(batch_size=2 ** 4)
    for batch in loader:
        print(batch["titles"])
        break

if __name__ == "__main__":
    main()
