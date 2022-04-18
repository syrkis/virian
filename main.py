# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, Wiki, lang_splits, get_args, get_params, train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from labml import experiment


# call stack
def main():

    args     = get_args()
    exp_name = 'test' if args.local else 'bsc'
    params   = get_params(args)

    if args.train:
        with experiment.record(name=exp_name, exp_conf=params):
            device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model     = Model(params); model.to(device) # get model and give to device
            ds        = Dataset(params['Languages'], params)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters())
            model     = train(ds, model, optimizer, criterion, device, params)

    if args.dataset:
        pass # ds = Dataset(langs)
    if args.wiki:
        pass # wiki = Wiki(langs)
    if args.ess:
        pass # ess = ESS()


if __name__ == "__main__":
    main()

