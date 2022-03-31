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
def train(loader, model, optimizer, criterion, month_count=(11 * 2500 * 5)//hypers['batch_size']): # langs * days * epochs
    with tqdm(islice(loader, month_count), total=month_count) as days:
        for X in days:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, X)
            loss.backward()
            optimizer.step()
            days.set_postfix(loss=loss.item())
    get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

