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

            # train batch
            optimizer.zero_grad()
            x_pred, y_pred = model(X, W)
            x_loss         = criterion(y_pred, Y)
            y_loss         = criterion(x_pred, X)

            # valid batch
            X_val, W_val, Y_val    = next(valid_iter)
            x_val_pred, y_val_pred = model(X_val, W_val)
            x_val_loss             = criterion(x_val_pred, X_val)
            ess_loss               = criterion(y_val_pred, Y_val)

            # backward
            wiki_loss = x_loss + x_val_loss
            loss      = wiki_loss + ess_loss
            loss.backward()
            optimizer.step()

            # report
            tracker.save(step, {'wiki mse': wiki_loss.item() / params["Batch Size"],
                                'ess mse': ess_loss.item() / params["Batch Size"]})

    return model


# make k fold loaders
def k_fold(ds, lang, params, device):
    t_idx, v_idx  = ds.k_fold(lang)
    train_sampler = SubsetRandomSampler(t_idx)
    valid_sampler = SubsetRandomSampler(v_idx)
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

