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
    for k, v in tqdm(groups.items()):
        group = df.loc[v][vals + ["essround"]].astype(int)
        descr = group.describe(datetime_is_numeric=True), group.mode()
        plt.plot(group["essround"], group[vals])
        plt.savefig(f'plots/{k}.png')
        plt.set_title(k)
        plt.clf()
        
