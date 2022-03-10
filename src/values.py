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
    ess_dir = "../data/dumps/ess/r_7_8_9_rel_sub"
    df = pd.read_csv(f"{ess_dir}/ESS1-9e01_1.csv", dtype='object', usecols=vals + meta).dropna()
    groups = df.groupby("cntry").groups
    countries = df.groupby("cntry").groups
    for idx, (k1, v1) in tqdm(enumerate(countries.items())):
        country = df.loc[v1]
        rounds = country.groupby("essround").groups
        modes = pd.DataFrame(columns = vals)
        for jdx, (k2, v2) in enumerate(rounds.items()):
            mode = country.loc[v2].mode()
            modes = pd.concat((modes, mode.dropna()))
        ys = modes[vals].T
        x = np.array(modes["essround"])
        for i in range(len(ys)):
            y = np.array(ys.iloc[i])
            plt.plot(x, y, label=vals[i])
        plt.legend()
        plt.savefig(f"plots/{k1}")
        plt.clf()

