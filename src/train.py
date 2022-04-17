#   virian training
# by: Noah Syrkis

# imports
from src.utils import parameters
import pickle
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


# train function
def train(ds, model, optimizer, criterion, device, writer):
    model.to(device)
    with trange(1, len(ds.langs), desc="all folds") as folds:
        for fold, lang in zip(folds, ds.langs):
            train_loader, valid_iter = k_fold(ds, lang)
            with tqdm(train_loader, desc="this fold") as t_fold:
                for step, (X, W, Y) in enumerate(t_fold):

                    # train
                    model.train()
                    X, W, Y = X.to(device), W.to(device), Y.to(device)
                    optimizer.zero_grad()
                    x_pred, y_pred = model(X, W)
                    x_loss, y_loss = criterion(x_pred, X), criterion(y_pred, Y)
                    loss = x_loss + y_loss
                    loss.backward()
                    optimizer.step()

                    # validate
                    model.eval()
                    X_val, W_val, Y_val = next(valid_iter)
                    X_val, W_val, Y_val = X_val.to(device), W_val.to(device), Y_val.to(device)
                    x_val_pred, y_val_pred = model(X_val, W_val)
                    x_val_loss, y_val_loss = criterion(x_val_pred, X_val), criterion(y_val_pred, Y_val)
    
                    # report
                    writer.add_scalars(f"Wiki Loss", {"train": x_loss.item(), "valid": x_val_loss.item()}, step)
                    writer.add_scalars(f"ESS Loss", {"train": y_loss.item(), "valid": y_val_loss.item()}, step)
                    t_fold.set_postfix(loss=loss.item())

    # get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")
    return model


# make k fold loaders
def k_fold(ds, lang):
    train_idx, val_idx           = ds.k_fold(lang)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    train_loader  = DataLoader(dataset=ds, batch_size=parameters['batch_size'], sampler=train_sampler, drop_last=True)
    valid_loader  = DataLoader(dataset=ds, batch_size=parameters['batch_size'], sampler=valid_sampler, drop_last=True)
    valid_iter    = cycle(valid_loader)
    return train_loader, valid_iter


def cycle(seq):
    while True:
        for X, W, Y in seq:
            yield X, W, Y


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

