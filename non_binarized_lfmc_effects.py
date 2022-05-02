# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:35:56 2022

@author: kkrao
"""

import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import init
import seaborn as sns
import statsmodels.api as sm
import statsmodels
import sklearn.metrics
import sklearn.linear_model
import scipy.special

df = pd.read_csv(os.path.join(init.dir_root, "data","r","matched_extreme_3_mar_2022.csv"))
df = df.rename(columns = {"y":"fire"})
# df.shape
# df.columns
# len(df["match"].unique())
# df.match.plot()
# df.z_high.mean()
def flatten(t):
    return [item for sublist in t for item in sublist]

def reshape(row):
    cols = ["lfmc","fire","lc","mean_ndvi","vpd_4m","agb","mean_t","wind"]
    select = [[f"{col}_0", f"{col}_1"] for col in cols]
    select = flatten(select)
    if len(row) == 2:
        order = np.argsort(row.lfmc)
        new_row = [list(np.array(row[col])[order]) for col in cols]
        new_row = flatten(new_row)
        new_row = pd.Series(data = new_row, index = select)
        # new_row = pd.Series(data = list(np.array(row.lfmc)[order]) +\
        #                             list(np.array(row.y)[order]) +\
        #                             list(np.array(row.lc)[order]),\
        #                     index = ["lfmc_0","lfmc_1","fire_0","fire_1", "lc_0","lc_1"])
        return new_row
    else:
        return pd.Series(data = [np.nan]*len(select), \
                         index = select)
cols = ["lfmc","fire","lc","mean_ndvi","vpd_4m","agb","mean_t","wind"]    
mdf = df.groupby("match")[cols].apply(reshape)
mdf["lfmc_diff"] = mdf["lfmc_0"] - mdf["lfmc_1"]
mdf.dropna(inplace = True)

# check if matches belong to same class.
# (mdf["lc_0"] == mdf["lc_1"]).mean()
mdf["thresh"] = list(map(init.thresh["extreme"].get, mdf.lc_0))
mdf.head()

mdf.fire_0.mean()

fig, ax = plt.subplots()
sns.kdeplot(x = "lfmc_1",data =  mdf,alpha = 0.5,\
            ax = ax, label = r"LFMC$_1$")
sns.kdeplot(x = "lfmc_0",data =  mdf,alpha = 0.5,\
            ax = ax, label = r"LFMC$_0$")
ax.set_xlabel("LFMC (%)")
ax.legend()

fig, ax = plt.subplots()
sns.kdeplot(x = "lfmc_diff",data =  mdf.loc[(mdf.fire_0==0) & (mdf.fire_1==0)],alpha = 0.5,\
            ax = ax, label = r"No fire")
sns.kdeplot(x = "lfmc_diff",data =  mdf.loc[(mdf.fire_0==1) & (mdf.fire_1==0)],alpha = 0.5,\
            ax = ax, label = r"Fire in dry region")
sns.kdeplot(x = "lfmc_diff",data =  mdf.loc[(mdf.fire_0==0) & (mdf.fire_1==1)],alpha = 0.5,\
            ax = ax, label = r"Fire in wet region")
ax.set_xlabel("$\Delta$ LFMC (%)")
ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")


fig, axs = plt.subplots(3, 1, figsize = (3, 2), sharex = True)
sns.boxplot(x = "lfmc_diff",data =  mdf.loc[(mdf.fire_0==0) & (mdf.fire_1==0)],\
            ax = axs[0], color = "C0")

sns.boxplot(x = "lfmc_diff",data =  mdf.loc[(mdf.fire_0==1) & (mdf.fire_1==0)],\
            ax = axs[1], color = "C1")
sns.boxplot(x = "lfmc_diff",data =  mdf.loc[(mdf.fire_0==0) & (mdf.fire_1==1)],\
            ax = axs[2], color = "C2")
axs[0].set_xlabel("")
axs[1].set_xlabel("")

axs[2].set_xlabel("$\Delta$ LFMC (%)")


mdf.loc[(mdf.fire_0==0) & (mdf.fire_1==1)].shape
#%% cms for low, med, high LFMC thresh

def get_or_stats(df):
    
    table = sklearn.metrics.confusion_matrix(\
                     np.append(df.fire_0.values,df.fire_1.values), \
                     np.append(np.repeat(1,len(df)),np.repeat(0,len(df))))
    model = statsmodels.stats.contingency_tables.Table2x2(table)

    return [model.oddsratio, model.oddsratio_confint()[1],  model.oddsratio_confint()[0], model.oddsratio_pvalue(), 2*len(df)]
    

or_by_lfmc = pd.DataFrame(index = init.thresh_abs.keys(), columns = ["Odds ratio", "upper","lower","p","n"])

for cat in init.thresh_abs.keys():
    sdf = mdf.loc[(mdf.lfmc_0 > init.thresh_abs[cat][0])&(mdf.lfmc_0 <= init.thresh_abs[cat][1])].copy()
    or_by_lfmc.loc[cat,:] = get_or_stats(sdf)

fig, ax = plt.subplots(figsize = (3,3))

ax.errorbar(or_by_lfmc.index, or_by_lfmc["Odds ratio"], \
            yerr = [or_by_lfmc["Odds ratio"] - or_by_lfmc["lower"],or_by_lfmc["upper"] - or_by_lfmc["Odds ratio"]], color = "grey", \
                capsize = 3, zorder = -1)
ax.scatter(or_by_lfmc.index, or_by_lfmc["Odds ratio"], color = "k", edgecolor = "grey")
ax.set_ylim(0,12)
ax.axhline(1, linestyle = "--",color = "grey")
ax.set_ylabel("Odds ratio")
ax.set_xlabel("Min. LFMC in matched pair")
new_labs = [f"({init.thresh_abs[cat][0]}, {init.thresh_abs[cat][1]}]" for cat in init.thresh_abs.keys()]
# ax.set_xlabel("Range of min. LFMC")
ax.set_xticklabels(new_labs)

print(or_by_lfmc)

#%% logistic regression

sdf = mdf.loc[(mdf.fire_0==1) |(mdf.fire_1==1) ].copy()
fig, ax= plt.subplots(figsize = (3,3))

ax.scatter(sdf.lfmc_0,sdf.fire_0, edgecolor = "C1", color = "None", linewidth = 0.5)
ax.scatter(sdf.lfmc_1,sdf.fire_1, edgecolor = "C0", color = "None", linewidth = 0.5)

ax.set_xlabel("LFMC (%)")
ax.set_ylabel("")
ax.set_yticks([0,1])
ax.set_yticklabels(["Unburned","Burned"])

clf = sklearn.linear_model.LogisticRegression()
X = np.append(sdf.lfmc_0.values,sdf.lfmc_1.values)
X = X.reshape((len(X), 1))
y = np.append(sdf.fire_0.values,sdf.fire_1.values)

clf.fit(X,y)
X_test = np.linspace(40, 200, 300)
loss = scipy.special.expit(X_test * clf.coef_ + clf.intercept_).ravel()
ax.plot(X_test, loss, color="black", linewidth=1)

fig, ax= plt.subplots(figsize = (3,3))
xaxis = "lfmc"
yaxis = "wind"
ax.scatter(sdf[f"{xaxis}_0"],sdf[f"{yaxis}_0"], c = sdf["fire_0"])
ax.scatter(sdf[f"{xaxis}_1"],sdf[f"{yaxis}_1"], c = sdf["fire_1"])
ax.set_xlabel(xaxis)
ax.set_ylabel(yaxis)

