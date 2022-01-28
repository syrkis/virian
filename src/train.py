# train.py
#   virian training
# by: Noah Syrkis

# imports
from tqdm import tqdm
from itertools import islice


# train function
def train(loader, model, optimizer, criterion, batch_count=2 ** 7):
    with tqdm(islice(loader, batch_count), unit="batch", total=batch_count) as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            _, pred = model(batch)
            loss = criterion(pred, batch)
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())


def main():
    pass

if __name__ == "__main__":
    main()
