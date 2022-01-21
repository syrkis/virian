# main.py
#   runs (and trains) virian NLP analytics
# by: Noah Syrkis

# imports
from src import Dataset
from src import Model
from src import train


# call stack
def main():
    ds = Dataset('da')
    loader = ds.pytorch(shuffle=True, batch_size= 2 ** 10)
    

if __name__ == "__main__":
    main()
