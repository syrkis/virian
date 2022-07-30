# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, Wiki, ESS, train, utils
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import random; random.seed(0)
from datetime import datetime


# call stack
def main():
    args        = utils.get_args()
    conf        = utils.get_conf()
    train_langs, test_langs = utils.train_test_split(list(conf['langs'].keys()))
    conf['train_langs'] = train_langs
    conf['test_langs'] = test_langs
    params = utils.get_params(args)

    if args.ess:
        ess = ESS(conf)

    if args.wiki:
        wiki = Wiki(conf)
        # wiki.get_dailies()
        # wiki.get_texts()
        wiki.text_to_vec()

    if args.dataset:
        ds = Dataset(conf, False) # False for test
        for i in range(5):
            print(ds[i])

    if args.model:
        model    = Model(params)
        train_ds = Dataset(conf)
        loader = DataLoader(dataset=train_ds, batch_size=16)
        for x_true, w_true, y_true in loader:
            x_pred, y_pred = model(x_true, w_true)
            print(y_pred.shape, x_pred.shape)
            break

    if args.train:
        now       = datetime.now().strftime("%Y-%m-%d-%H-%M")
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds        = Dataset(conf)
        fold_size = len(ds.langs) // 5
        for fold, i in enumerate(range(0, len(ds.langs), fold_size)):
            langs                    = ds.langs[i: i + fold_size] 
            train_loader, valid_iter = utils.cross_validate(ds, langs, params, device)
            model                    = Model(params); model.to(device);
            criterion                = nn.MSELoss()
            optimizer                = optim.Adam(model.parameters(), lr=params['Learning Rate'])
            train(train_loader, valid_iter, model, optimizer, criterion, params, fold, now)


if __name__ == "__main__":
    main()

