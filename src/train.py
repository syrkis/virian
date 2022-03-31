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
def train(loader, model, optimizer, criterion): # langs * days * epochs
    with tqdm(loader) as data:
        for X, W in data:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, X)
            loss.backward()
            optimizer.step()
            data.set_postfix(loss=loss.item())
    get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

