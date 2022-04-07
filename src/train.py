# train.py
#   virian training
# by: Noah Syrkis

# imports
import pickle
from tqdm import tqdm
from itertools import islice
from src.utils import get_s3, hypers
import datetime
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


# train function
def train(ds, model, optimizer, criterion, writer, idx=0):
    for epoch, lang in enumerate(ds.langs):
        train_loader, val_loader = k_fold(ds, lang)
        with tqdm(train_loader) as tepoch:
            for X, W, Y in tepoch:
                optimizer.zero_grad()
                x_pred, y_pred = model(X, W, Y)
                loss_x = criterion(x_pred, X)
                # loss_y = criterion(y_pred, Y)
                writer.add_scalar("Wiki Train Loss", loss_x, idx:= idx + 1)
                # writer.add_scalar("ESS Loss", loss_y, idx)
                loss = loss_x # + loss_y
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            with torch.no_grad():
                validate(val_loader, model, criterion, writer, idx)
    return model


# compuate epoch validation score
def validate(loader, model, criterion, writer, idx):
    for X, W, Y in loader:
        x_pred, y_pred = model(X, W, Y)
        loss_x = criterion(x_pred, X)
        writer.add_scalar("Wiki Val Loss", loss_x, idx:= idx + 1)


# make k fold loaders
def k_fold(ds, lang):
    train_idx, val_idx = ds.k_fold(lang)
    train_sampler      = SubsetRandomSampler(train_idx)
    val_sampler        = SubsetRandomSampler(val_idx)
    train_loader       = DataLoader(dataset=ds, batch_size=hypers['batch_size'], sampler=train_sampler)
    val_loader         = DataLoader(dataset=ds, batch_size=hypers['batch_size'], sampler=val_sampler)
    return train_loader, val_loader


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

