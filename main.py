# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, Wiki, ESS, train, utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# call stack
def main():
    args = utils.get_args()
    conf = utils.get_conf()

    if args.ess:
        ess = ESS(conf)

    if args.dataset:
        ds = Dataset(conf)
        for i in range(5):
            print(ds[i][-1])

    if args.model:
        model = Model(params)
        

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
            langs                    = ds.langs[i:i+fold_size] 
            train_loader, valid_iter = utils.cross_validate(ds, langs, params, device)
            model                    = Model(params); model.to(device);
            criterion                = nn.MSELoss()
            optimizer                = optim.Adam(model.parameters(), lr=params['Learning Rate'])
            train(train_loader, valid_iter, model, optimizer, criterion, params, fold)

    if args.wiki:
        wiki = Wiki(params)
        wiki.texts_to_toks(params['Vocab Size'])
        # wiki.get_dailies_lang('sl')
        # wiki.get_texts_lang('sl') # TODO: FIX SL text file (no hash)
        wiki.text_to_vec() # recompute vector representation of summaries


if __name__ == "__main__":
    main()

