# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:33:04 2020

@author: kkrao
"""


import pandas as pd
from init import dir_data, lc_dict, color_dict
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm




sns.set(style='ticks',font_scale = 0.9)

def assemble_df(trait =  "p50"):
    df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
    
    dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_extra_lfmc_vars_500m_variogram.csv"))
    dfr = dfr[['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside']]
    df = df.join(dfr)
    
    dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_fwi_500m_variogram.csv"))
    dfr = dfr[['fwi_t_4_inside','fwi_t_4_outside']]
    df = df.join(dfr)
    
    dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_500m_with_p50.csv"))
    dfr = dfr[[trait]]
    df = df.join(dfr)
    
    df = df.loc[df.landcover.isin(lc_dict.keys())]
    df['landcover'] = df.landcover.map(lc_dict)
    return df


def plot_p50_hist(df):
    fig, ax = plt.subplots(figsize = (3,3))
    df.p50.hist(ax = ax)
    ax.set_xlabel('P50 (Mpa)')
    ax.set_ylabel('Frequency')

# plot_p50_hist(df)

def custom_round(x, base=2):
    return int(base * round(float(x)/base))


def myround(x, base=2):
    return np.ceil(x / float(base)) * base -1
#%% just lfmc first 
def auc_by_p50(clf, ndf, kind = 'occurence', step = 2, trait = "p50", trait_min = -12, trait_max = 1):
    auc = pd.DataFrame(index = [0],columns = np.arange(trait_min,trait_max, step))
    for p in auc.columns:
        sub = ndf.loc[ndf[trait].apply(lambda x: custom_round(x, step)).astype(int)==p]
        if kind == "occurence":
            X = sub.drop(['fire',trait], axis = 1)
            y = sub['fire']
            auc.loc[0,p] = sklearn.metrics.roc_auc_score(y, clf.predict(X))
        else:
            X = sub.drop(['size',trait], axis = 1)
            y = sub['size']
            try:
                auc.loc[0,p] = sklearn.metrics.roc_auc_score(y, clf.predict(X))
            except:
                auc.loc[0,p] = np.nan
    return auc


def calc_auc_size(dfsub, clf):
    ndf = dfsub.copy()
    ndf = ndf.sample(frac=1).reset_index(drop=True)
    ndf.dropna(inplace = True)
    # print(ndf.columns)
    X = ndf.drop(['size','p50'], axis = 1)
    y = ndf['size']
    # print(y.mean())
    # try:
    clf.fit(X, y)
        # rfc_disp = sklearn.metrics.plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
    auc = auc_by_p50(clf, ndf, kind = "size")
        # print(sklearn.metrics.roc_auc_score(y, clf.predict(X)))
    # except: 
        # print("Could not fit RF")
    # print(auc)        
    return auc

def calc_auc_occurence(dfsub,  clf, trait = "p50"):
    df = dfsub.copy()
    
    ndf = pd.DataFrame()
    for var in ['outside','inside']:    
        cols = [col for col in df.columns if var in col]+[trait]
        # cols.remove('lfmc_t_1_%s'%var)
        data = df[cols].copy()
        new_cols = [col.split('_')[0] for col in data.columns]
        data.columns = (new_cols)
        data['fire'] = int(var=='inside')
        ndf = pd.concat([ndf, data], axis = 0).reset_index(drop=True)
        

    ndf = ndf.sample(frac=1).reset_index(drop=True)
    ndf.dropna(inplace = True)

    X = ndf.drop(['fire',trait], axis = 1)
    y = ndf['fire']
    
    try:
        clf.fit(X, y)
        # rfc_disp = sklearn.metrics.plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
        auc = auc_by_p50(clf, ndf, kind = "occurence")
        return auc
    except: 
        print("Could not fit RF")
        return np.nan
    # return auc

def ensemble_auc(dfsub, clf, iters = 5, kind = "occurence"):
    clf.random_state = 0
    # dummy = calc_auc_occurence(dfsub, category_dict, clf)
    # aucs = np.expand_dims(dummy.values, axis = 2)
    for itr in range(1, iters):
        clf.random_state = itr
        if itr ==1:
            if kind == 'occurence':
                auc = calc_auc_occurence(dfsub, clf)
            else:
                auc = calc_auc_size(dfsub, clf)
        else:
            if kind == 'occurence':
                auc = auc.append(calc_auc_occurence(dfsub, clf)).reset_index(drop=True)
            else: 
                auc = auc.append(calc_auc_size(dfsub, clf)).reset_index(drop=True)
        # aucs = np.append(aucs,auc, axis = 2)
    # print("aucs ready")
    # dummy.loc[:,:] = np.nanmean(aucs.astype(float), axis = 2)
    # mean = dummy.copy()
    # dummy.loc[:,:] = np.nanstd(aucs.astype(float), axis = 2)
    # sd = dummy.copy()

    return auc    

def calc_auc_diff(dfs, replace_by_random = False, kind = "occurence",trait = "p50"):
    df = dfs.copy()
    # allVars = pd.DataFrame(index = [0],columns = category_dict.keys())
    # onlyClimate = allVars.copy()
    cols = [col for col in df.columns if 'lfmc' in col]+[trait,'area']
    # cols = ['landcover']
    cols+=[col for col in df.columns if 'erc' in col]
    cols+=[col for col in df.columns if 'ppt' in col]
    cols+=[col for col in df.columns if 'vpd' in col]
    cols+=[col for col in df.columns if 'fwi' in col]
    
    df = df[cols]
    df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside'] - df['lfmc_t_1_seasonal_mean_inside']
    df['lfmc_t_1_outside_anomaly'] = df['lfmc_t_1_outside'] - df['lfmc_t_1_seasonal_mean_outside']
    
    df['size'] = 0
    df.loc[df.area>4,'size'] = 1
    
    df.drop('area',axis = 1, inplace = True)
    
    df.drop(['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside'],axis = 1, inplace = True)
    
    if kind == 'size':
        remove_outside = [col for col in df.columns if "outside" in col]
        df.drop(remove_outside, axis = 1,inplace = True)
    ###testing with random numbers instead of LFMC
    # df.loc[:,remove_lfmc] = np.zeros(shape = df.loc[:,remove_lfmc].shape)
    # clf = RandomForestClassifier(max_depth=15, min_samples_leaf = 5, random_state=0, oob_score = True,n_estimators = 50)
    
    if kind=='occurence':
        clf = RandomForestClassifier(max_depth=6, random_state=0, oob_score = True,n_estimators = 20)
    else:
        clf = RandomForestClassifier(max_depth=10, min_samples_leaf = 1, random_state=0, oob_score = True,n_estimators = 20)
    allVars = ensemble_auc(df, clf, kind = kind)
    
    
    # allVars = calc_auc(df, size_dict, clf)

    remove_lfmc = [col for col in df.columns if 'lfmc' in col]
    if replace_by_random:
        ###testing with random numbers instead of LFMC
        df.loc[:,remove_lfmc] = np.ones(shape = df.loc[:,remove_lfmc].shape)
        onlyClimate, s2 = ensemble_auc(df, clf, kind = kind)
    else:
        onlyClimate = ensemble_auc(df.drop(remove_lfmc, axis = 1), clf, kind = kind)
    
    diff = (allVars - onlyClimate).copy().astype(float).round(3)
    onlyClimate.index.name = "only climate"
    diff.index.name = "difference, mean"
    allVars.index.name = "all variables"
    
    # sd = (s1.pow(2)+s2.pow(2)).pow(0.5).astype(float).round(3)
    # sd.index.name = "difference, sd"
    # print(onlyClimate.astype(float).round(2))
    # print(allVars.astype(float).round(2))
    # print(diff.astype(float).round(2))
    # print(sd.astype(float).round(2))
    
    return allVars, onlyClimate

def plot_importance(allVars, onlyClimate, xlab = 'P50 (MPA)'):
    y = (allVars - onlyClimate)/onlyClimate*100
    mean = y.mean()
    sd = y.std()
    
    fig, ax= plt.subplots(figsize = (3,3))
    ax.plot(mean,'ko-', markeredgecolor = "grey")
    ax.errorbar(x = mean.index, y = mean, yerr = sd, fmt = 'o', color = "grey", capsize = 2, zorder = -1)
    ax.scatter(x = mean.index, y = mean,marker = 'o', edgecolor = "grey")


    ax.set_ylabel("LFMC Value (%)")
    ax.set_xlabel(xlab)
    # ax.set_xlim(1,-15)
    # ax.set_xticks([0,-5,-10,-15])
    # ax1.set_xlim(0.5,1)
    # ax1.set_title("Small fires")

df = assemble_df()
allVars, onlyClimate = calc_auc_diff(df, replace_by_random = True, kind = 'occurence')
plot_importance(allVars, onlyClimate)
# plot_importance(mean, std, onlyClimate, replace_by_random = False)
# print(mean)
# print(std)
