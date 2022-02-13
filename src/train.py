# train.py
#   virian training
# by: Noah Syrkis

# imports
import pickle
from tqdm import tqdm
from itertools import islice
from src.helpers import get_s3


# train function
def train(ds, model, optimizer, criterion, month_count=80):
    with tqdm(islice(ds, month_count), unit="batch", total=month_count) as month:
        for X, W, Y in month:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, X)
            loss.backward()
            optimizer.step()
            month.set_postfix(loss=loss.item())
    get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

