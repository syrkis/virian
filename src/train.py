# train.py
#   virian training
# by: Noah Syrkis

# imports
from src.utils import cycle, get_metrics

import torch

from tqdm import tqdm
import wandb


# train function
def train(train_loader, valid_iter, model, optimizer, criterion, params, fold):
    wandb.init(entity='syrkis', project='bsc', job_type='train', name=f'fold_{fold}',
            config=params, reinit=True, group="full data")
    wandb.watch(model)
    for idx, (X, W, Y) in enumerate(train_loader):
        optimizer.zero_grad()

        # train set
        model.train()
        x_pred, y_pred = model(X, W)
        x_loss         = criterion(x_pred, X)
        y_loss         = criterion(y_pred, Y)

        # validation set
        model.eval()
        X_val, W_val, Y_val    = next(valid_iter)
        x_pred_val, y_pred_val = model(X_val, W_val) 
        x_loss_val             = criterion(x_pred_val, X_val)
        y_loss_val             = criterion(y_pred_val, Y_val)

        # report and evluate
        train_acc = torch.sum((y_pred > 0) == (Y > 0)) / torch.numel(Y)
        valid_acc = torch.sum((y_pred_val > 0) == (Y_val > 0)) / torch.numel(Y_val)
        metrics   = get_metrics(x_loss, y_loss, x_loss_val, y_loss_val, train_acc, valid_acc, params)
        wandb.log(metrics)

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

