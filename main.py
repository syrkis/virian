# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, Wiki, ESS, train, utils
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


# call stack
def main():
    args   = utils.get_args()
    conf   = utils.get_conf()
    params = utils.get_params(args)

    if args.ess:
        ess = ESS(conf)

    if args.wiki:
        wiki = Wiki(conf)
        wiki.text_to_vec()

    if args.dataset:
        ds = Dataset(conf)
        for i in range(5):
            print(ds[i])

    if args.model:
        model  = Model(params)
        ds     = Dataset(conf)
        loader = DataLoader(dataset=ds, batch_size=16)
        for x_true, w_true, y_true in loader:
            x_pred, y_pred = model(x_true, w_true)
            print(y_pred.shape, x_pred.shape)
            break

    if args.train:
        device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds           = Dataset(conf)
        fold_size    = len(ds.langs) // 5
        for fold, i in enumerate(range(0, len(ds.langs), fold_size)):
            langs                    = ds.langs[i: i + fold_size] 
            train_loader, valid_iter = utils.cross_validate(ds, langs, params, device)
            model                    = Model(params); model.to(device);
            criterion                = nn.MSELoss()
            optimizer                = optim.Adam(model.parameters(), lr=params['Learning Rate'])
            train(train_loader, valid_iter, model, optimizer, criterion, params, fold)


if __name__ == "__main__":
    main()

