# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset, Model, Wiki, lang_splits, get_args, train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def run_train(train_langs, test_langs):
    model     = Model()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds        = Dataset(train_langs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    writer    = SummaryWriter()
    model     = train(ds, model, optimizer, criterion, device, writer)
    writer.flush()


# call stack
def main():
    args = get_args()

    if args.ess:
        ess = ESS()

    if args.wiki:
        train_langs, test_langs = lang_splits.values()
        all_langs = train_langs + test_langs
        # wiki = Wiki(all_langs)
        # wiki.get_dailies() # don't run

    if args.train:
        train_langs, test_langs = lang_splits.values()
        if args.local:
            run_train(train_langs[:2], test_langs)
        run_train(train_langs, test_langs)

    if args.dataset:
        train_langs, _ = lang_splits.values()
        if args.local:
            ds = Dataset(train_langs[:2])
        shapes = []
        for X, W, Y in tqdm(ds):
            shapes.append(X.shape)

if __name__ == "__main__":
    main()

