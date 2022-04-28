# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, Wiki, ESS, train, utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from labml import experiment
import wandb
from tqdm import tqdm


# call stack
def main():

    args   = utils.get_args()
    params = utils.get_params(args)
    langs  = params['Languages']

    if args.model:
        ds     = Dataset(params)
        loader = DataLoader(dataset=ds, batch_size=params["Batch Size"])
        model  = Model(params)
        for X, W, Y in tqdm(loader):
            x_pred, y_pred = model(X, W)

    if args.train:
        exp_name     = 'local' if args.local else 'bsc'
        device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds           = Dataset(params)
        fold_size    = len(ds.langs) // 5
        for fold, i in enumerate(range(0, len(ds.langs), fold_size)):
            wandb.init(entity='syrkis', project='bsc', job_type='train', name=f'fold_{fold}')
            wandb.config = params
            langs = ds.langs[i:i+fold_size] 
            # with experiment.record(name=exp_name, exp_conf=params): # from lambml days
            train_loader, valid_iter = utils.cross_validate(ds, langs, params, device)
            model                    = Model(params); model.to(device); wandb.watch(model)
            criterion                = nn.MSELoss()
            optimizer                = optim.Adam(model.parameters(), lr=params['Learning Rate'])
            train(train_loader, valid_iter, model, optimizer, criterion, params)

    if args.dataset:
        ds = Dataset(params)
        shapes = []
        for X, W, Y in tqdm(ds):
            shapes.append((X.shape, W.shape, Y.shape))

    if args.wiki:
        wiki = Wiki(params)
        # wiki.texts_to_toks(params['Vocab Size'])
        # wiki.get_dailies_lang('sl')
        # wiki.get_texts_lang('sl')
        # wiki.text_to_vec() # recompute vector representation of summaries

    if args.ess:
        ess = ESS()
        out = ess.get_human_values("fi", "2019_01_10")


if __name__ == "__main__":
    main()

