# ess.py
#   virian ess analysis
# by: Noah Syrkis

# imports
from src.utils import variables
from datetime import datetime, timedelta

import torch
from torch import tensor
import pandas as pd
import numpy as np

# from factor_analyzer import FactorAnalyzer as FA
from matplotlib import pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.cluster import KMeans
from collections import Counter


# make values
class ESS:
    ess_file = f"data/ess/ESS8e02_2-ESS9e03_1-ESS10.csv"

    def __init__(self, conf):
        self.conf   = conf
        self.Y      = self.construct()
        self.df     = self.get_df(conf)
        self.rounds = self._make_rounds()
        self.ranges = self.get_country_date_ranges()
        # self.descrete()     # edits self.avg and self.std
        # self.avg, self.std = self.get_avg_std(conf)

    def construct(self):
        get_avg = lambda w, v: np.sum((w * v.T).T, axis=0) / np.sum(w)
        get_var = lambda w, v: np.sum((w * ((v - get_avg(w, v)) ** 2).T).T, axis=0) / np.sum(w)
        vals = list(self.conf['cols']['values'].keys())
        meta = self.conf['cols']['meta']
        cols = vals + meta
        ess          = pd.read_csv(self.ess_file, dtype='object', usecols=cols)
        ess          = ess[ess['dweight'].notna()]
        ess['cntry'] = ess['cntry'].apply(lambda x: x.lower())
        groups       = ess.groupby(['cntry', 'essround']).groups
        Y = {cntry : {'avg':{},'var':{}} for cntry in list(set(ess['cntry']))}
        for k, v in groups.items():
            group   = ess.iloc[v]
            values  = (group[vals].to_numpy()).astype(np.int64)
            weights = (group['dweight'].to_numpy()).astype(np.float64)
            avg     = get_avg(weights, values) # weigthed means
            var     = get_var(weights, values)
            Y[k[0]]['avg'][k[1]] = avg
            Y[k[0]]['var'][k[1]] = var
        C = [self.conf['langs'][k] for k in self.conf['train_langs']]
        Y = self.z_score(Y, C)
        return Y


    def z_score(self, Y, C):
        avg_avg = np.zeros(21)
        avg_var = np.zeros(21)
        print(C)
        for c in C:
            avg_avg += list(Y[c]['avg'].values())[0] / len(C)
            avg_var += list(Y[c]['var'].values())[0] / len(C)
        
        var_avg = np.zeros(21)
        var_var = np.zeros(21)
        for c in C:
            var_avg += ((list(Y[c]['avg'].values())[0] - avg_avg) ** 2) / len(C)
            var_var += ((list(Y[c]['var'].values())[0] - avg_var) ** 2) / len(C)

            
        for k1, v1 in Y.items():
            for k2, v2 in v1['avg'].items():
                Y[k1]['avg'][k2] = ((v2 - avg_avg) / var_avg ** .5).tolist()
            for k2, v2 in v1['var'].items():
                Y[k1]['var'][k2] = ((v2 - avg_var) / var_var ** .5).tolist()
        return Y

    def descrete(self):
        out = []
        fig, axes = plt.subplots(2, 1, figsize=(5, 8))
        for idx, df in enumerate([self.avg, self.std]):
            out.append(self.get_cluster(df, axes[idx]))
        # plt.show() # amazing plot (use in report)
        self.avg = out[0]
        self.std = out[1]
        print(out[0].value_counts().idxmax())

    def get_cluster(self, df, ax):
        df2 = df.copy()
        for idx, col in enumerate(self.conf['cols']['values']):
            df2[col] = pd.cut(df2[col], bins=3, labels=False)
        return df2 - 1

    def get_country_date_ranges(self):
        ranges = {cntry : {} for cntry in set(self.df.cntry)} # list of length 1 to 3 of tuples with round, dates
        groups = self.df.groupby(['essround', 'cntry']).groups
        for k, v in groups.items():
            group = self.df.loc[v]
            _range = self.get_date_range(k[0], group)
            if _range:
                ranges[k[1]][k[0]] = _range
        return ranges

    def get_date_range(self, _round, df):
        if _round == 10:
            date = pd.to_datetime(df['inwds'])
            date = date.dropna()
            min_date = min(date)
            max_date = max(date)
        else:
            date = df[['inwyys', 'inwmms', 'inwdds']]
            date.columns = ['year', 'month', 'day']
            date = date[(date.year < 2022) & (date.year >= 2015) & (date.month < 13) & (date.day != 31)]
            date = pd.to_datetime(date)
            date = date.dropna()
            if len(date) == 0:
                return False
            min_date = min(date)
            max_date = max(date)
        return (min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))

    def get_df(self, conf):
        countries   = list(conf['langs'].values())
        df          = pd.read_csv(self.ess_file, low_memory=False)
        df['cntry'] = df['cntry'].str.lower()
        df          = df.loc[df['cntry'].isin(countries)]
        return df


    def get_avg_std(self, conf):
        target = conf['cols']['values']
        cols   = list(target.keys()) + conf['cols']['meta']
        groups = self.df.groupby(["cntry", "essround"]).groups
        avg    = pd.DataFrame({col: [0.0] * len(groups) for col in target}, index=groups.keys())
        std    = pd.DataFrame({col: [0.0] * len(groups) for col in target}, index=groups.keys())
        for k, group in groups.items():
            group = self.df.loc[group] # loc or iloc??
            for col in target:
                data  = group.where(group[col] <= target[col])
                stats = DescrStatsW(data[col], weights=data['pspwght'])
                avg.loc[k][col] = data[col].mean()
                std.loc[k][col] = data[col].std()
        avg = (avg - np.mean(avg, axis=0)) / np.std(avg, axis=0)
        std = (std - np.mean(std, axis=0)) / np.std(std, axis=0)
        return avg, std

    def get_target(self, lang, date):
        country   = self.conf['langs'][lang]
        ess_round = str(self._date_to_round(country, date))
        avg = self.Y[country]['avg'][ess_round]
        var = self.Y[country]['var'][ess_round]
        Y = tensor([avg, var]).T.float()
        return Y

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
        round_dates   = [(7,"2014"),(8,"2016"),(9,"2018"),(10,"2020")]
        round_to_date = {k: (self._to_date(f'{v}_12_31'), k) for k, v in round_dates}
        keys          = self.df.groupby(['cntry', 'essround']).groups.keys()
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
