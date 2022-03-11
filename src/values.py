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


# values analysis
def load_df():
    meta = "essround cntry".split()
    vals = "IPUDRST IPMODST IPGDTIM IMPFREE IPHLPPL IPSUCES IPSTRGV IPADVNT IPBHPRP IPRSPOT IPLYLFR IMPENV IMPTRAD IMPFUN".lower().split()
    df = pd.read_csv("../data/ess/data.csv", dtype='object', usecols=vals + meta).dropna()
    df = df[vals].astype(int)
    fa = FactorAnalyzer(n_factors=5, rotation=None) 
    df = fa.fit_transform(df)
    print(df.var(axis=0))
    print(df.mean(axis=0))
    sns.heatmap(fa.loadings_)
    plt.show()
    exit()
    countries = df.groupby("cntry").groups
    for k1, v1 in countries.items():
        country = df.loc[v1]
        ess_rounds = country.groupby("essround").groups
        for k2, v2 in ess_rounds.items():
            ess_round = (country.loc[v2][vals]).apply(pd.to_numeric)
            print(ess_round[vals].var(axis=0))

