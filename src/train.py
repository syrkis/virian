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
def train(loader, model, optimizer, criterion, epochs=5): # langs * days * epochs
    for epoch in range(epochs):
        with tqdm(loader) as tepoch:
            for X, W, Y in tepoch:
                optimizer.zero_grad()
                x_pred, y_pred = model(X, W, Y)
                print(y_pred)
                loss = criterion(x_pred, X)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        # get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

