# train.py
#   virian training
# by: Noah Syrkis

# imports
from tqdm import tqdm


# train function
def train(loader, model, optimizer, criterion):
    for epoch in range(1, 10):
        with tqdm(loader, unit=" batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch in tepoch:
                optimizer.zero_grad()
                pred = model(batch)
                loss = criterion(pred, batch)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())


def main():
    pass

if __name__ == "__main__":
    main()
