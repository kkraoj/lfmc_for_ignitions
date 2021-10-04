# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:33:04 2020

@author: kkrao
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble
import sklearn.metrics
import statsmodels.api as sm

import init

sns.set(style='ticks',font_scale = 0.9)

def assemble_df():
    df = pd.read_csv(os.path.join(init.dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
    
    dfr = pd.read_csv(os.path.join(init.dir_data, "fire_collection_median_extra_lfmc_vars_500m_variogram.csv"))
    dfr = dfr[['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside']]
    df = df.join(dfr)
    
    dfr = pd.read_csv(os.path.join(init.dir_data, "fire_collection_median_fwi_500m_variogram.csv"))
    dfr = dfr[['fwi_t_4_inside','fwi_t_4_outside']]
    df = df.join(dfr)
    
    dfr = pd.read_csv(os.path.join(init.dir_data, "fire_collection_500m_with_p50.csv"))
    dfr = dfr[['p50']]
    df = df.join(dfr)
    
    df = df.loc[df.landcover.isin(init.lc_dict.keys())]
    df['landcover'] = df.landcover.map(init.lc_dict)
    return df


def plot_p50_hist(df):
    fig, ax = plt.subplots(figsize = (3,3))
    df.p50.hist(ax = ax)
    ax.set_xlabel('P50 (Mpa)')
    ax.set_ylabel('Frequency')



#%% just lfmc first 


def auc_by_cat(clf, ndf, ndf_dict, kind = 'occurence'):
    auc = pd.DataFrame(index = [0],columns = ndf_dict.keys())
    for cat in auc.columns:
        sub = ndf.loc[ndf_dict[cat]].copy()
        if kind == "occurence":
            X = sub.drop(['fire'], axis = 1)
            y = sub['fire']
            # auc.loc[0,cat] = sklearn.metrics.roc_auc_score(y, clf.predict(X))
            auc.loc[0,cat] = sklearn.metrics.roc_auc_score(y, clf.oob_decision_function_.argmax(axis = 1)[ndf_dict[cat]])
        else:
            X = sub.drop(['size'], axis = 1)
            y = sub['size']
            try:
                auc.loc[0,cat] = sklearn.metrics.roc_auc_score(y, clf.oob_decision_function_.argmax(axis = 1)[ndf_dict[cat]])
            except:
                auc.loc[0,cat] = np.nan
    return auc

def calc_auc_size(dfsub, clf):
    df = dfsub.copy()
    auc = pd.DataFrame(index = sorted(df.landcover.unique()),columns = ['auc'])
    for lc in sorted(df.landcover.unique()):
        ndf = df.loc[df.landcover==lc].copy()
      
        # ndf['size'] = 0
        # ndf.loc[ndf.area>4,'size'] = 1
        ndf = ndf.sample(frac=1).reset_index(drop=True)
        ndf.dropna(inplace = True)
        # print(ndf.columns)
        X = ndf.drop(['size', 'area', "landcover"], axis = 1)
        y = ndf['size']
        # print(y.mean())
        try:
            clf.fit(X, y)
            # rfc_disp = sklearn.metrics.plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
            auc.loc[lc,'auc'] = sklearn.metrics.roc_auc_score(y, clf.predict(X))
            # print(sklearn.metrics.roc_auc_score(y, clf.predict(X)))
        except: 
            print("Could not fit RF for land cover: %s"%(lc))
    # print(auc)        
    return auc

def calc_auc_occurence(dfsub, category_dict, category, clf):
    df = dfsub.copy()
    # auc = pd.DataFrame(index = [0],columns = category_dict.keys())

    ndf = pd.DataFrame()
    
    for var in ['outside','inside']:    
        cols = [col for col in df.columns if var in col] + [category]
        # cols.remove('lfmc_t_1_%s'%var)
        data = df[cols].copy()
        new_cols = [col.split('_')[0] for col in data.columns]
        data.columns = (new_cols)
        data['fire'] = int(var=='inside')
        ndf = pd.concat([ndf, data], axis = 0).reset_index(drop=True)
        

    ndf = ndf.sample(frac=1).reset_index(drop=True)
    ndf.dropna(inplace = True)
    
    ndf_dict = {}
    for i in ndf[category].unique():
        ndf_dict[i] = ndf[category]==i
    ndf = ndf.drop([category], axis = 1)
    X = ndf.drop(['fire'], axis = 1)
    y = ndf['fire']
    
    try:
        clf.fit(X, y)
        # rfc_disp = sklearn.metrics.plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
        auc = auc_by_cat(clf, ndf, ndf_dict, kind = 'occurence')
        return auc
    except: 
        print("Could not fit RF")
        auc = pd.DataFrame(index = [0],columns = ndf_dict.keys())
        auc.loc[:,:] = np.nan
        return auc

def ensemble_auc(dfsub, category_dict, category, clf, iters = 100, label = 'All variables'):
    clf.random_state = 0
    dummy = calc_auc_occurence(dfsub, category_dict, category, clf)
    aucs = np.expand_dims(dummy.values, axis = 2)
    for itr in range(1, iters):
        clf.random_state = itr
        auc = np.expand_dims(calc_auc_occurence(dfsub, category_dict, category, clf).values, axis = 2)
        
        aucs = np.append(aucs,auc, axis = 2)
    # print("aucs ready")
    dummy.loc[:,:] = np.nanmean(aucs.astype(float), axis = 2)
    mean = dummy.copy()
    dummy.loc[:,:] = np.nanquantile(aucs.astype(float), 0.05,  axis = 2)
    ql = (dummy.copy() - mean).abs() ## for plotting we need absolute
    dummy.loc[:,:] = np.nanquantile(aucs.astype(float), 0.95,  axis = 2)
    qu = dummy.copy() - mean
    return mean, ql, qu
    

def calc_auc_diff(dfs, category_dict, category, replace_by_random = False):
    df = dfs.copy()
    allVars = pd.DataFrame(index = [0],columns = category_dict.keys())
    onlyClimate = allVars.copy()
    cols = [col for col in df.columns if 'lfmc' in col]+['landcover']
    # cols = ['landcover']
    cols+=[col for col in df.columns if 'erc' in col]
    cols+=[col for col in df.columns if 'ppt' in col]
    cols+=[col for col in df.columns if 'vpd' in col]
    cols+=[col for col in df.columns if 'fwi' in col]
    
    df = df[cols]
    df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside'] - df['lfmc_t_1_seasonal_mean_inside']
    df['lfmc_t_1_outside_anomaly'] = df['lfmc_t_1_outside'] - df['lfmc_t_1_seasonal_mean_outside']
    
    df.drop(['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside'],axis = 1, inplace = True)
        
    ###testing with random numbers instead of LFMC
    # df.loc[:,remove_lfmc] = np.zeros(shape = df.loc[:,remove_lfmc].shape)
    # clf = sklearn.ensemble.RandomForestClassifier(max_depth=15, min_samples_leaf = 5, random_state=0, oob_score = True,n_estimators = 50)
    clf = sklearn.ensemble.RandomForestClassifier(max_depth=None, random_state=0, oob_score = True,n_estimators = 50)

    allVars, ql, qu = ensemble_auc(df, category_dict,category, clf)
    
    
    # allVars = calc_auc(df, size_dict, clf)

    remove_lfmc = [col for col in df.columns if 'lfmc' in col]
    if replace_by_random:
        ###testing with random numbers instead of LFMC
        df.loc[:,remove_lfmc] = np.random.uniform(size = df.loc[:,remove_lfmc].shape)
        onlyClimate, _, _ = ensemble_auc(df,category_dict, category, clf)
    else:
        onlyClimate, _, _ = ensemble_auc(df.drop(remove_lfmc, axis = 1), category_dict,category, clf)
    
    diff = (allVars - onlyClimate).copy().astype(float).round(3)
    onlyClimate.index.name = "only climate"
    diff.index.name = "difference, mean"
    allVars.index.name = "all variables"
    
    # sd = (s1.pow(2)+s2.pow(2)).pow(0.5).astype(float).round(3)
    ql.index.name = "difference, ql"
    qu.index.name = "difference, qu"
    # print(onlyClimate.astype(float).round(2))
    # print(allVars.astype(float).round(2))
    # print(diff.astype(float).round(2))
    # print(sd.astype(float).round(2))
    
    df = pd.DataFrame({"all_vars":allVars.iloc[0], "only_climate":onlyClimate.iloc[0], "lower_q": ql.iloc[0], "upper_q":qu.iloc[0]})
    df = df.sort_values("all_vars", ascending = True)

    return df

def plot_importance(df):
    fig, ax1= plt.subplots(figsize = (3,3))
    
    ax1.barh(width = df.only_climate,y = df.index,edgecolor = list(df.index.map(init.color_dict).values), color = "w")

    ax1.barh(width = df.all_vars - df.only_climate,y = df.index,left = df.only_climate, \
             color = list(df.index.map(init.color_dict).values), \
             edgecolor = list(df.index.map(init.color_dict).values),\
                 xerr = df[['lower_q', 'upper_q']].T.values,\
                     )

    ax1.set_ylabel("")
    ax1.set_xlabel('AUC')
    
    ax1.set_xlim(0.5,1)
    # ax1.set_title("Small fires")

##############################################################################

df = assemble_df()
df = df.loc[df.BurnDate>=150]
# df = df.loc[df.area>=1]
LC_DICT = {}
for i in df.landcover.unique():
    LC_DICT[i] = df.landcover==i
P50_DICT = {'low':(df.p50>=-5),
             'high': (df.p50<-5),     
             }
# plot_p50_hist(df)

auc = calc_auc_diff(df, LC_DICT, category = "landcover", replace_by_random = True)

plot_importance(auc)
# print(mean)
# print(std)

##############################################################################
#%%invs

# for lc in LC_DICT.keys():
#     sub = df.loc[df.landcover==lc]
#     fig, ax = plt.subplots(figsize = (3,3))
#     ax.hist(sub.lfmc_t_2_inside, color = "orange", alpha = 0.5, density = True, label = "Inside", bins = 100)
#     ax.hist(sub.lfmc_t_2_outside, color = "blue", alpha = 0.5, density = True, label = "Outside", bins = 100)
#     ax.set_xlabel("LFMC_t_1")
#     ax.set_ylabel("Frequency")
#     plt.show()

