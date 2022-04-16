# ess.py
#   virian ess analysis
# by: Noah Syrkis

# imports
from src.utils import variables, lang_to_country
from datetime import datetime, timedelta

import torch
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer as FA


# make values
class ESS:

    nans     = [66,77,88,99]
    facts    = 'a b c d e'.split()
    ess_file = f"{variables['data_dir']}/ess/ESS1-9e01_1.csv"

    def __init__(self):
        self.raw                     = pd.read_csv(self.ess_file, usecols=meta+data)
        self.raw                     = self.raw.replace(self.nans, np.NaN) # 99 etc. to NaN
        self.raw['cntry']            = self.raw['cntry'].str.lower()
        self.raw_avg, self.raw_var   = self._make_summary()
        self.fact_avg, self.fact_var = self._make_factors()
        self.rounds                  = self._make_rounds()

    def get_target(self, lang, date, factor=True):
        country   = lang_to_country[lang]
        ess_round = self._date_to_round(country, date)
        avg       = self.fact_avg.loc[country, ess_round]
        var       = self.fact_var.loc[country, ess_round]
        Y         = torch.from_numpy(np.array((avg.to_numpy(), var.to_numpy()))).float()
        return Y

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


meta = """essround,cntry""".split(',') # add pspwght
data = """health,hlthhmp,rlgblg,rlgdnm,rlgblge,rlgdnme,rlgdgr,rlgatnd,pray,happy,sclmeet,inprdsc,sclact,crmvct,aesfdrk,ipcrtiv,imprich,ipeqopt,ipshabt,impsafe,impdiff,ipfrule,ipudrst,ipmodst,ipgdtim,impfree,iphlppl,ipsuces,ipstrgv,ipadvnt,ipbhprp,iprspot,iplylfr,impenv,imptrad,impfun""".split(',')


def main():
    pass

if __name__ == "__main__":
    main()
