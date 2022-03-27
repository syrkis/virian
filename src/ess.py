# ess.py
#   virian ess analysis
# by: Noah Syrkis

# imports
import pandas as pd
from src.utils import paths
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer


# make values
def construct_factors():
    meta_cols = "essround cntry".split()
    ess_data = pd.read_csv(paths['ess'], dtype='object', usecols=meta_cols + val_cols)
    fa = FactorAnalyzer(n_factors=5, rotation="promax")
    fa.fit(ess_data[val_cols])
    countries = ess_data.groupby('cntry').groups
    D = {}

    for k1, v1 in countries.items():
        D [k1] = {}
        country = ess_data.loc[v1]
        rounds = country.groupby("essround").groups

        for k2, v2 in rounds.items():
            D[k1][k2] = {}
            round_factors = fa.transform(country.loc[v2][val_cols].dropna().astype(int))
            D[k1][k2]['var'] = round_factors.var(axis=0)
            D[k1][k2]['avg'] = round_factors.mean(axis=0)

    return D

# helpers
val_cols = "ipcrtiv imprich ipeqopt ipshabt impsafe impdiff ipfrule ipudrst ipmodst ipgdtim impfree iphlppl ipsuces ipstrgv ipadvnt ipbhprp iprspot iplylfr impenv imptrad impfun".split()

