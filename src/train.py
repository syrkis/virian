#   virian training
# by: Noah Syrkis

# imports
from src.utils import cycle

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

from labml import tracker


# train function
def train(ds, model, optimizer, criterion, device, params):
    for fold, lang in enumerate(ds.langs):
        train_loader, valid_iter = k_fold(ds, lang, params, device)
        for step, (X, W, Y) in enumerate(train_loader):

            optimizer.zero_grad()

            x_pred, y_pred = model(X, W) # predict on train data
            x_loss, y_loss = criterion(x_pred, X), criterion(y_pred, Y)

            X_val, W_val, Y_val    = next(valid_iter) # val data
            x_pred_val, y_pred_val = model(X_val, W_val) 
            x_loss_val, y_loss_val = criterion(x_pred_val, X_val), criterion(y_pred_val, Y_val)

            tracker.save(step, {'wiki mse': x_loss.item() / params["Batch Size"],
                'ess train mse': y_loss.item() / params["Batch Size"],
                'ess valid mse': y_loss_val.item() / params["Batch Size"]})

            loss = x_loss + y_loss + x_loss_val
            loss.backward()

            optimizer.step()

    return model


# make k fold loaders
def k_fold(ds, lang, params, device):
    train_idx, valid_idx  = ds.k_fold(lang)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader  = get_loader(ds, params['Batch Size'], train_sampler, device)
    valid_loader  = get_loader(ds, params['Batch Size'], valid_sampler, device)
    valid_iter    = cycle(valid_loader)
    return train_loader, valid_iter


def get_loader(ds, batch_size, sampler, device):
    loader = DataLoader(dataset=ds, batch_size=batch_size, sampler=sampler, drop_last=True,
             collate_fn=lambda x: list(map(lambda x: x.to(device), default_collate(x))))
    return loader


def to_device(x, w, y, device):
    x = default_collate(x).to(device)
    w = default_collate(w).to(device)
    y = default_collate(y).to(device)
    return x, w, y

# call stack
def main():
    pass

if __name__ == "__main__":
    main()

