# train.py
#   virian training
# by: Noah Syrkis

# imports
from src.utils import cycle, get_metrics

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

from tqdm import tqdm
from labml import tracker


# train function
def train(train_loader, valid_iter, model, optimizer, criterion, params):
    for idx, (X, W, Y) in enumerate(train_loader):
        optimizer.zero_grad()

        # train set
        x_pred, y_pred = model(X, W)
        x_loss         = criterion(x_pred, X)
        y_loss         = criterion(y_pred, Y)

        # validation set
        X_val, W_val, Y_val    = next(valid_iter)
        x_pred_val, y_pred_val = model(X_val, W_val) 
        x_loss_val             = criterion(x_pred_val, X_val)
        y_loss_val             = criterion(y_pred_val, Y_val)

        # report and evluate
        train_acc = torch.numel((y_pred > 0) == (Y > 0))
        valid_acc = torch.numel((y_pred_val > 0) == (Y_val > 0))
        tracker.save(idx, get_metrics(x_loss, y_loss, y_loss_val, x_loss_val, train_acc, valid_acc))

        # backpropagate and update weights
        loss = x_loss + y_loss
        loss.backward()
        optimizer.step()
    return model


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

