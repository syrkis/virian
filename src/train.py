#   virian training
# by: Noah Syrkis

# imports
from src.utils import parameters
import pickle
from tqdm import tqdm, trange
from itertools import cycle
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from memory_profiler import profile


# train function
@profile
def train(ds, model, optimizer, criterion, device, writer, idx=0):
    model.to(device)
    with trange(1, len(ds.langs), desc="all folds") as folds:
        for fold, lang in zip(folds, ds.langs):
            train_loader, val_iter = k_fold(ds, lang)
            with tqdm(train_loader, desc="this fold") as fold:
                for X, W, Y in fold:
                    
                    # tensors to GPU
                    X, W, Y = X.to(device), W.to(device), Y.to(device)
              
                    # clean tensor gradients
                    optimizer.zero_grad()

                    # make predictions
                    x_pred, y_pred = model(X, W)

                    # calcualte loss
                    x_loss, y_loss = criterion(x_pred, X), criterion(y_pred, Y)

                    # report loss
                    loss = x_loss + y_loss

                    # backpropagate errors
                    loss.backward()

                    # update model parameters
                    optimizer.step()

                    # run validation
                    val_loss = validate(val_iter, model, criterion, device)

                    # write loss to tqdm
                    report_loss(loss, x_loss, y_loss, val_loss, writer, fold, idx:=idx+1)

    # get_s3().put_object(Bucket="models", Body=pickle.dumps(model.state_dict()), Key="model.pth.pkl")
    return model


# compute and report loss
def report_loss(loss, x_loss, y_loss, val_loss, writer, fold, idx):
    writer.add_scalars(f"Wiki Loss", {"training": x_loss, "validation": val_loss[0]}, idx)
    writer.add_scalars(f"ESS Loss", {"training": y_loss, "validation": val_loss[1]}, idx)
    fold.set_postfix(loss=loss.item())


# compuate epoch validation score
def validate(val_iter, model, criterion, device):
    with torch.no_grad():
        X, W, Y        = next(val_iter)
        X, W, Y        = X.to(device), W.to(device), Y.to(device)
        x_pred, y_pred = model(X, W)
        x_loss, y_loss = criterion(x_pred, X), criterion(y_pred, Y)
        return x_loss, y_loss


# make k fold loaders
def k_fold(ds, lang):
    train_idx, val_idx = ds.k_fold(lang)
    train_sampler      = SubsetRandomSampler(train_idx)
    val_sampler        = SubsetRandomSampler(val_idx)
    train_loader       = DataLoader(dataset=ds, batch_size=parameters['batch_size'], sampler=train_sampler)
    val_loader         = DataLoader(dataset=ds, batch_size=parameters['batch_size'], sampler=val_sampler)
    return train_loader, cycle(val_loader)


# test
def test(ds, model):
    with torch.no_grad():
        pass


# call stack
def main():
    pass

if __name__ == "__main__":
    main()

