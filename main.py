# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, Wiki, ESS, train, utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from labml import experiment
from tqdm import tqdm


# call stack
def main():

    args   = utils.get_args()
    params = utils.get_params(args)
    langs  = params['Languages']

    if args.model:
        ds     = Dataset(params['Languages'], params)
        loader = DataLoader(dataset=ds, batch_size=2)
        model  = Model(params)
        for X, W, _ in tqdm(loader):
            x_pred = model(X, W)

    if args.train:
        exp_name = 'local' if args.local else 'bsc'
        device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds       = Dataset(params['Languages'], params)
        for fold, lang in enumerate(ds.langs):
            with experiment.record(name=exp_name, exp_conf=params):
                train_loader, valid_iter = utils.cross_validate(ds, lang, params, device)
                model = Model(params)
                model.to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=params['Learning Rate'])
                train(train_loader, valid_iter, model, optimizer, criterion, params)

    if args.dataset:
        ds = Dataset(params)
    if args.wiki:
        wiki = Wiki(params)
        # wiki.texts_to_toks(params['Vocab Size'])
        wiki.text_to_vec()
    if args.ess:
        ess = ESS()
        ess.base_model()


if __name__ == "__main__":
    main()

