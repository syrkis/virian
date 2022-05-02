# ess.py
#   virian ess analysis
# by: Noah Syrkis

# imports
from src.utils import variables, lang_to_country, ess_cols
from scipy.stats import zscore
from datetime import datetime, timedelta

import torch
from torch import tensor
import pandas as pd
import numpy as np

from factor_analyzer import FactorAnalyzer as FA
from matplotlib import pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW



# make values
class ESS:

    nans     = [66,77,88,99]
    facts    = 'a b c d e'.split()
    ess_file = f"{variables['data_dir']}/ess/ESS1-9e01_1.csv"

    def __init__(self, params):
        self.data = self.precompute(params)
        exit()
        self.raw                     = self.raw.replace(self.nans, np.NaN) # 99 etc. to NaN
        self.raw['cntry']            = self.raw['cntry'].str.lower()
        self.raw_avg, self.raw_var   = self._make_summary()
        self.fact_avg, self.fact_var = self._make_factors()
        self.rounds                  = self._make_rounds()

    def precompute(self, params):
        print(ess_cols[params['Target']])
        cols   = ess_cols[params['Target']] + ess_cols['meta']
        nats   = [lang_to_country[lang].upper() for lang in params['Languages']]
        df     = pd.read_csv(self.ess_file, usecols=cols)
        groups = df.groupby(["cntry", "essround"]).groups
        avg    = pd.DataFrame({col: [0.0] * len(groups) for col in ess_cols[params['Target']]}, index=groups.keys())
        var    = pd.DataFrame({col: [0.0] * len(groups) for col in ess_cols[params['Target']]}, index=groups.keys())
        for k, group in groups.items():
            group = df.iloc[group].dropna()
            for col in ess_cols[params['Target']]:
                data  = group.where(group[col] <= 10).dropna()
                stats = DescrStatsW(data[col], weights=data['pspwght'])
                avg.loc[k][col] = stats.mean
                var.loc[k][col] = stats.std
        print(avg.std())
        print(var.std())
        return avg, var
        # avg.to_csv('data/avg.csv', index=False)
        # var.to_csv('data/var.csv', index=False)


    def get_target_fact(self, lang, date):
        country   = lang_to_country[lang]
        ess_round = self._date_to_round(country, date)
        avg       = self.fact_avg.loc[country, ess_round]
        var       = self.fact_var.loc[country, ess_round]
        Y         = torch.from_numpy(np.array((avg.to_numpy(), var.to_numpy()))).float()
        return Y

    def get_human_values(self, lang, date):
        avg       = zscore(self.raw_avg, axis=0)
        var       = zscore(self.raw_var, axis=0)
        country   = lang_to_country[lang]
        # a = np.mean(np.sum(avg > 0) / avg.shape[0])
        # b = np.mean(np.sum(var > 0) / var.shape[0])
        # print(1 - (a +  b)/ 2) # baseline correctness
        ess_round = self._date_to_round(country, date)
        out_avg   = avg.loc[country].loc[float(ess_round)][ess_cols["human_values"]].tolist()
        out_var   = var.loc[country].loc[float(ess_round)][ess_cols["human_values"]].tolist()
        return tensor([out_avg, out_var]).T # make numbers have same scale as emebddings
   
    def base_model(self):
        avg = self.fact_avg
        var = self.fact_var
        print(avg.mean())
        print(var.mean())

    def _date_to_round(self, country, date):
        date   = datetime.strptime(date, "%Y_%m_%d")
        rounds = self.rounds[country]
        pos    = np.argmin([abs(r[0] - date) for r in rounds])
        return rounds[pos][1]
        
    def _make_rounds(self):
        round_dates   = [(7, "2014"), (8, "2016"), (9, "2018")]
        round_to_date = {k: (self._to_date(f'{v}_12_31'), k) for k, v in round_dates}
        keys          = self.raw.groupby(['cntry', 'essround']).groups.keys()
        rounds        = {k: [] for k, _ in keys}
        for k, v in keys: # (country, round)
            rounds[k].append(round_to_date[v])
        return rounds

    def _to_date(self, string):
        return datetime.strptime(string, "%Y_%m_%d") 

    def _make_summary(self):
        groups           = self.raw.groupby(['cntry', 'essround'])
        raw_avg, raw_var = groups.mean(), groups.var()
        raw_avg = raw_avg.loc[self.countries] # remove countries not in focus
        raw_var = raw_var.loc[self.countries] # remove countries not in focus
        return raw_avg, raw_var

    def _make_factors(self):
        avg_fa = FA(n_factors=5, rotation="promax")
        var_fa = FA(n_factors=5, rotation="promax")
        avg_fa.fit(self.raw_avg)
        var_fa.fit(self.raw_var)
        fact_avg = pd.DataFrame(avg_fa.transform(self.raw_avg),
                columns=self.facts, index=self.raw_avg.index)
        fact_var = pd.DataFrame(var_fa.transform(self.raw_var),
                columns=self.facts, index=self.raw_var.index)
        return fact_avg, fact_var


meta = """essround,cntry""".split(',') # add pspwght weight!!!
data = """health,hlthhmp,rlgblg,rlgdnm,rlgblge,rlgdnme,rlgdgr,rlgatnd,pray,happy,sclmeet,inprdsc,sclact,crmvct,aesfdrk,ipcrtiv,imprich,ipeqopt,ipshabt,impsafe,impdiff,ipfrule,ipudrst,ipmodst,ipgdtim,impfree,iphlppl,ipsuces,ipstrgv,ipadvnt,ipbhprp,iprspot,iplylfr,impenv,imptrad,impfun""".split(',')


def main():
    pass

if __name__ == "__main__":
    main()
