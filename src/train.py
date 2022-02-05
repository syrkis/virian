# train.py
#   virian training
# by: Noah Syrkis

# imports
import pickle
from tqdm import tqdm
from itertools import islice
from src.helpers import get_s3


# train function
def train(loader, model, optimizer, topic_criterion, value_criterion=None, batch_count=5000):
    with tqdm(islice(loader, batch_count), unit="batch", total=batch_count) as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            x_pred, y_pred = model(batch)
            loss = topic_criterion(x_pred, batch)
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
    get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

