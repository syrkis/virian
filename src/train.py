# train.py
#   virian training
# by: Noah Syrkis

# imports
import torch
from tqdm import tqdm
import wandb
from src.utils import baseline


# train function
def train(train_loader, valid_iter, model, optimizer, criterion, params, fold):
    wandb.init(entity='syrkis', project='virian', job_type='train', name=f'fold_{fold}', config=params, reinit=True, group="full data")
    wandb.watch(model)
    idxs = len(train_loader.dataset) / params['Batch Size']
    for idx, (X, W, Y) in enumerate(train_loader):
        optimizer.zero_grad()

        # train set
        model.train()
        x_pred, y_pred = model(X, W)
        x_loss         = criterion(x_pred, X)
        y_loss         = criterion(y_pred, Y)
        predictions    = (y_pred > 0.5).long()

        # validation set
        model.eval()
        X_val, W_val, Y_val    = next(valid_iter)
        x_pred_val, y_pred_val = model(X_val, W_val)
        x_loss_val             = criterion(x_pred_val, X_val)
        y_loss_val             = criterion(y_pred_val, Y_val)

        # backpropagate and update weights
        loss = x_loss * ((idxs - idx)/idxs) + y_loss * (idx/idxs)
        loss.backward()
        optimizer.step()
        wandb.log({
            # "ESS Baseline MSE" : y_loss_val
            "ESS Baseline"     : baseline(y_pred_val, Y_val),
            "ESS Train MSE"    : y_loss.item(),
            "ESS Valid MSE"    : y_loss_val.item(),
            "Wiki Train MSE"   : x_loss.item(),
            "Wiki Valid MSE"   : x_loss_val.item(),
        })
    return model


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

