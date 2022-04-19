# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import (Dataset, Model, Wiki, ESS,
        lang_splits, get_args, get_params, train,
        cross_validate)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from labml import experiment


# call stack
def main():

    args     = get_args()

    if args.train:
        exp_name = 'local' if args.local else 'bsc'
        params   = get_params(args)
        device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds       = Dataset(params['Languages'], params)
        for fold, lang in enumerate(ds.langs):
            with experiment.record(name=exp_name, exp_conf=params):
                train_loader, valid_iter = cross_validate(ds, lang, params, device)
                model = Model(params)
                model.to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=params['Learning Rate'])
                train(train_loader, valid_iter, model, optimizer, criterion, params)

    if args.dataset:
        pass # ds = Dataset(langs)
    if args.wiki:
        pass # wiki = Wiki(langs)
    if args.ess:
        ess = ESS()
        ess.base_model()


if __name__ == "__main__":
    main()

