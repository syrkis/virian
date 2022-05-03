# train.py
#   virian training
# by: Noah Syrkis

# imports
from src.utils import cycle, get_metrics

import torch

from tqdm import tqdm
import wandb


# train function
def train(train_loader, valid_iter, model, optimizer, wiki_criterion, ess_criterion, params, fold):
    wandb.init(entity='syrkis', project='bsc', job_type='train', name=f'fold_{fold}', config=params, reinit=True, group="full data")
    wandb.watch(model)
    for idx, (X, W, Y) in enumerate(train_loader):
        optimizer.zero_grad()

        # train set
        model.train()
        x_pred, y_pred = model(X, W)
        x_loss         = wiki_criterion(x_pred, X)
        y_loss         = ess_criterion(y_pred, Y)
        predictions    = (y_pred > 0.5).long()
        ess_accuracy   = (y_pred == Y).sum() / torch.numel(Y)
        ess_acc        = (Y == (y_pred > 0.5).int()).int().sum() / torch.numel(Y)
        

        # validation set
        model.eval()
        X_val, W_val, Y_val    = next(valid_iter)
        x_pred_val, y_pred_val = model(X_val, W_val) 
        x_loss_val             = wiki_criterion(x_pred_val, X_val)
        y_loss_val             = ess_criterion(y_pred_val, Y_val)
        ess_accuracy_val       = (y_pred_val == Y_val).sum() / torch.numel(Y)
        ess_acc_val            = (Y_val == (y_pred_val > 0.5).int()).int().sum() / torch.numel(Y_val)

        # backpropagate and update weights
        loss = x_loss + y_loss
        loss.backward()
        optimizer.step()
        wandb.log({
            "ESS Train Accuracy" : ess_acc,
            "ESS Valid Accuracy" : ess_acc_val,
            "Wiki Train MSE"     : x_loss,
            "Wiki Valid MSE"     : x_loss_val
        })
    return model


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

