# main.py
#   describes articles relative to haidt dimensions  
# by: Noah Syrkis

# imports
import argparse
from src import Region

# get arguments
def get_args():
    parser = argparse.ArgumentParser(description='run virian') 
    parser.add_argument('--country', default=None, type=str)
    parser.add_argument('--wiki', default='en', type=str)
    parser.add_argument('--month', type=str)
    args = parser.parse_args()
    return args


# call stack
def main(): 
    args = get_args()
    region = Region(args.country)
    region.analyze(args.month)

if __name__ == '__main__':
    main()
