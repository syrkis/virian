# values.py
#   virian ess analysis
# by: Noah Syrkis

# imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


# values analysis
def load_df():
    meta = "essround cntry".split()
    vals = ["IPUDRST".lower()] #IPMODST IPGDTIM IMPFREE IPHLPPL IPSUCES IPSTRGV IPADVNT IPBHPRP IPRSPOT IPLYLFR IMPENV IMPTRAD IMPFUN".lower().split()
    df = pd.read_csv("../data/ess/data.csv", dtype='object', usecols=vals + meta).dropna()
    groups = df.groupby("cntry").groups
    countries = df.groupby("cntry").groups
     

