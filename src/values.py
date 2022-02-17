# values.py
#   virian ess analysis
# by: Noah Syrkis

# imports
import pandas as pd
from tqdm import tqdm
from factor_analyzer import FactorAnalyzer
from matplotlib import pyplot as plt


# values analysis
def load_df():
    meta = "essround cntry".lower().split()
    vals = "IPUDRST IPMODST IPGDTIM IMPFREE IPHLPPL IPSUCES IPSTRGV IPADVNT IPBHPRP IPRSPOT IPLYLFR IMPENV IMPTRAD IMPFUN".lower().split()
    ess_dir = "../data/dumps/ess/r_7_8_9_rel_sub"
    df = pd.read_csv(f"{ess_dir}/ESS1-9e01_1.csv", dtype='object', usecols=vals + meta).dropna()
    groups = df.groupby("cntry").groups
    countries = df.groupby("cntry").groups
    for idx, (k1, v1) in enumerate(countries.items()):
        country = df.loc[v1]
        rounds = country.groupby("essround").groups
        for val in vals:
            val = country[val]
        for jdx, (k2, v2) in enumerate(rounds.items()):
            round = country.loc[v2]
            descr = round.describe()[vals]
            plt.scatter(k2, descr)
            plt.savefig('plots/fuck.png')
            break
        break

