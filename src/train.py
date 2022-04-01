# train.py
#   virian training
# by: Noah Syrkis

# imports
import pickle
from tqdm import tqdm
from itertools import islice
from src.utils import get_s3, hypers
import datetime


# train function
def train(loader, model, optimizer, criterion, writer, epochs=5, idx=0): # langs * days * epochs
    for epoch in range(epochs):
        with tqdm(loader) as tepoch:
            for X, W, Y in tepoch:

                optimizer.zero_grad()
                x_pred, y_pred = model(X, W, Y)

                loss_x = criterion(x_pred, X)
                loss_y = criterion(y_pred, Y)

                writer.add_scalar("Wiki Loss", loss_x, idx:= idx + 1)
                writer.add_scalar("ESS Loss", loss_y, idx)

                loss = loss_x + loss_y
                loss.backward()

                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

    return model


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

