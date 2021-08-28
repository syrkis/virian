# runner.py
#   runs virian nlp
# by: Noah Syrkis

# imports
import torch
from dataset import Dataset
import os
import requests
import json


# evaluate a given date
def runner(date, local):
    ds = Dataset(date=date)


          

