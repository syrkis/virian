# values.py
#   virian ess analysis
# by: Noah Syrkis

# imports
import pandas as pd


# values analysis
def load_df():
    ess_dir = "../data/dumps/ess/r_7_8_9_rel_sub"
    df = pd.read_csv(f"{ess_dir}/ESS1-9e01_1.csv", dtype='object')
    print(df.head())


