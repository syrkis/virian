# values.py
#   virian ess analysis
# by: Noah Syrkis

# imports
import pandas as pd


# values analysis
def load_df():
    vals = "CNTRY INWMME IPUDRST IPMODST IPGDTIM IMPFREE IPHLPPL IPSUCES IPSTRGV IPADVNT IPBHPRP IPRSPOT IPLYLFR IMPENV IMPTRAD IMPFUN".lower().split()
    ess_dir = "../data/dumps/ess/r_7_8_9_rel_sub"
    df = pd.read_csv(f"{ess_dir}/ESS1-9e01_1.csv", dtype='object', usecols=vals)
    print(df.groupby(['cntry', "inwmme"]).groups)


