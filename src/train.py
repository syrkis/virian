# train.py
#   virian training
# by: Noah Syrkis

# imports
import torch
from tqdm import tqdm
import wandb


# train function
def train(train_loader, valid_iter, model, optimizer, criterion, params, fold):
    wandb.init(entity='syrkis', project='bsc', job_type='train', name=f'fold_{fold}', config=params, reinit=True, group="full data")
    wandb.watch(model)
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
        baseline               = criterion(torch.zeros_like(Y_val), Y_val)

        # backpropagate and update weights
        loss = x_loss + y_loss
        loss.backward()
        optimizer.step()
        wandb.log({
            # "ESS Baseline MSE" : y_loss_val
            "ESS Baseline MSE" : torch.sum(baseline).item() / torch.numel(Y_val),
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

