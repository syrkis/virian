# values.py
#   virian ess analysis
# by: Noah Syrkis

# imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer
import seaborn as sns


# make values
def make_values(langs):
    D = {lang: {} for lang in langs}
    lang2country = {'no':'NO', 'da':'DK'}
    country2lang = {'NO':'no', 'DK':'da'}
    meta_cols = "essround cntry".split()
    val_cols = "ipcrtiv imprich ipeqopt ipshabt impsafe impdiff ipfrule ipudrst ipmodst ipgdtim impfree iphlppl ipsuces ipstrgv ipadvnt ipbhprp iprspot iplylfr impenv imptrad impfun".split()
    ess_data = pd.read_csv('../data/ess/data.csv', dtype='object', usecols=meta_cols + val_cols)
    fa = FactorAnalyzer(n_factors=5, rotation="promax")
    fa.fit(ess_data[val_cols])
    countries = ess_data.groupby('cntry').groups
    for k1, v1 in countries.items():
        if k1 in country2lang:
            country = ess_data.loc[v1]
            rounds = country.groupby("essround").groups
            for k2, v2 in rounds.items():
                round_fa = fa.transform(country.loc[v2][val_cols].astype(int))
                round_my_sigma = (round_fa.mean(axis=0), round_fa.var(axis=0)) 
                D[country2lang[k1]][k2] = round_my_sigma
    D = [(k, v) for k, v in D.items()]
    return D


# values analysis
def factor_analysis(n_factors):
    meta = "essround cntry".split()
    vals = "ipcrtiv imprich ipeqopt ipshabt impsafe impdiff ipfrule ipudrst ipmodst ipgdtim impfree iphlppl ipsuces ipstrgv ipadvnt ipbhprp iprspot iplylfr impenv imptrad impfun".split()
    df = pd.read_csv("../data/ess/data.csv", dtype='object', usecols=vals + meta).dropna()
    fa = FactorAnalyzer(n_factors=n_factors, rotation="promax")
    fa.fit(df[vals])

    # _, v = fa.get_eigenvalues()
    countries = df.groupby("cntry").groups
    for k1, v1 in countries.items():
        country = df.loc[v1]
        ess_rounds = country.groupby("essround").groups
        for k2, v2 in ess_rounds.items():
            ess_round = (country.loc[v2][vals]).apply(pd.to_numeric)
            tmp = fa.transform(ess_round)
            print(tmp.mean(axis=0).round(2), tmp.var(axis=0).round(2))
        print()
