# ess.py
#   virian ess analysis
# by: Noah Syrkis

# imports
from src.utils import paths
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer as FA



# make values
class ESS:

    nans  = [66,77,88,99]
    facts = 'a b c d e'.split()

    def __init__(self):
        self.raw = pd.read_csv(paths['ess'], usecols=meta+data).replace(self.nans, np.NaN)
        self.raw_avg, self.raw_var   = self._make_summary()
        self.fact_avg, self.fact_var = self._make_factors()
        self.rounds = self._make_rounds()

    def get_target(self, country, date, factor=True):
        ess_round = self._date_to_round(country, date)
        avg = self.fact_avg.loc[country, date]
        var = self.fact_var.loc[country, date]
        return avg, var

    def _date_to_round(self, country, date):
        return self.rounds[country]
        
    def _make_rounds(self):
        keys = self.raw.groupby(['cntry', 'essround']).groups.keys()
        rounds = {k: set() for k, _ in keys}
        for k, v in keys:
            rounds[k].add(v)
        return rounds

    def _make_summary(self):
        groups = self.raw.groupby(['cntry', 'essround'])
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


# helpers
val_cols = "ipcrtiv imprich ipeqopt ipshabt impsafe impdiff ipfrule ipudrst ipmodst ipgdtim impfree iphlppl ipsuces ipstrgv ipadvnt ipbhprp iprspot iplylfr impenv imptrad impfun".split()


meta = """essround,cntry""".split(',') # add pspwght
data = """health,hlthhmp,rlgblg,rlgdnm,rlgblge,rlgdnme,rlgdgr,rlgatnd,pray,happy,sclmeet,inprdsc,sclact,crmvct,aesfdrk,ipcrtiv,imprich,ipeqopt,ipshabt,impsafe,impdiff,ipfrule,ipudrst,ipmodst,ipgdtim,impfree,iphlppl,ipsuces,ipstrgv,ipadvnt,ipbhprp,iprspot,iplylfr,impenv,imptrad,impfun""".split(',')

def main():
    ess = ESS()
    out = ess._date_to_round('SE', '2020_10_30')
    print(out)


if __name__ == "__main__":
    main()
