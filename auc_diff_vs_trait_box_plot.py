# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 23:14:16 2021

@author: kkrao
"""


import os
import itertools  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import statsmodels.api as sm

import init


def assemble_df(buffer_size = 10000, trait = "p50_liu"):
    df = pd.read_csv(os.path.join(init.dir_data, "varying_buffers", \
                    f"fire_collection_median_with_climate_fwi_extra_lfmc_vars_constant_buffer_{buffer_size:d}_width_10km.csv"))
    # dfr = pd.read_csv(os.path.join(init.dir_data, "fire_collection_500m_with_p50.csv"))
    dfr = pd.read_csv(os.path.join(init.dir_data, "fires_with_traits_only_2016_2019_13_oct_2021.csv"))
    dfr = dfr[['p50','isohydricity','p50_liu','root_depth','hft']]
    df = df.join(dfr)
    
    df = df.loc[df.landcover.isin(init.lc_dict.keys())]
    df['landcover'] = df.landcover.map(init.lc_dict)
    
    df = df.rename(columns = {'isohydricity':'sigma',"root_depth":"rootdepth","p50_liu":"p50liu"})
      
    df["p50"] = pd.qcut(df.p50.round(1), 5).astype(str)
    df["sigma"] = pd.qcut(df.sigma.round(1), 5).astype(str)
    df["p50liu"] = pd.qcut(df.p50liu.round(0), 5).astype(str)
    df["rootdepth"] = pd.qcut(df.rootdepth.round(1), 5).astype(str)
    # df["hft"] = pd.qcut(df.hft, 6)

    
    columns = ['system:index', 'BurnDate', 'FirstDay', 'LastDay', 'QA', 'Uncertainty',
    'area', 'erc_t_15_inside', 'erc_t_15_outside',
    'landcover', 'lfmc_t_1_inside', 'lfmc_t_1_outside', 'ppt_t_1_inside',
    'ppt_t_1_outside', 'system:time_start', 'vpd_t_4_inside',
    'vpd_t_4_outside', 'year', '.geo', 'lfmc_t_1_seasonal_mean_inside',
    'lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside',
    'fwi_t_4_inside', 'fwi_t_4_outside',trait]
    df = df[columns]
    df.dropna(inplace = True)


    return df

def make_x_y(df, trait = "landcover"):
    ndf = pd.DataFrame()
    
    for var in ['outside','inside']:    
        cols = [col for col in df.columns if var in col] + [trait]
        # cols.remove('lfmc_t_1_%s'%var)
        data = df[cols].copy()
        new_cols = [col.split('_')[0] for col in data.columns]
        data.columns = (new_cols)
        data['fire'] = int(var=='inside')
        ndf = pd.concat([ndf, data], axis = 0).reset_index(drop=True)
        

    ndf = ndf.sample(frac=1).reset_index(drop=True)
    X = ndf.drop([trait, 'fire'], axis = 1).values
    y = ndf['fire'].values
    
    return X, y, ndf
    
    
def get_true_preds(df, kf, clf, trait = "landcover"):
    X, y, x_y = make_x_y(df, trait = trait)
    x_y["pred"] = 0
    for index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        x_y.loc[test_index, "pred"] = clf.predict(X_test)
    
    return x_y[["fire","pred",trait]]    

def get_combo(rf_seed, cv_seed, buffer_size, max_depth, folds = 3, trait = "landcover"):
    
    df = assemble_df(buffer_size = buffer_size, trait = trait)
    kf = sklearn.model_selection.KFold(n_splits=folds, shuffle = False, random_state = cv_seed)
    clf = sklearn.ensemble.RandomForestClassifier(max_depth=max_depth, random_state=rf_seed, oob_score = True,n_estimators = 50)
    
    return df, kf, clf

def get_auc_by_trait(_true_preds, trait = "landcover"):
    auc = pd.Series(index = sorted(_true_preds[trait].unique()), dtype = "float")
    for cat in auc.index:
        auc.loc[cat] = sklearn.metrics.roc_auc_score(\
                                                       _true_preds.loc[_true_preds[trait]==cat, "fire"],\
                                                           _true_preds.loc[_true_preds[trait]==cat, "pred"])

    return auc

def remove_lfmc(df):
    remove_lfmc = [col for col in df.columns if 'lfmc' in col]
    df.loc[:,remove_lfmc] = np.random.uniform(size = df.loc[:,remove_lfmc].shape)
    return df

def combine_results_into_frame(allVars, onlyClimate, trait = "landcover"):
    allVars = allVars.melt().rename(columns = {"value":"auc", "variable":trait})
    allVars["vars"]="LFMC+Met."
    onlyClimate = onlyClimate.melt().rename(columns = {"value":"auc", "variable":trait})
    onlyClimate["vars"]="Met."

    return allVars.append(onlyClimate, ignore_index=True)
    
def box_plot_lc(to_plot):
    
    fig, ax = plt.subplots(figsize = (10,5))
    ax = sns.boxplot(x="landcover", y="auc", hue="vars",
                 data=to_plot, palette="Set3", ax =ax)
    ax.set_xlabel("")

def box_plot_trait(diff, trait = "landcover"):
    to_plot = diff.melt()
    to_plot = to_plot.rename(columns = {"variable":trait,"value":"auc_diff"})
    
    fig, ax = plt.subplots(figsize = (5,5))
    ax = sns.boxplot(x=trait, y="auc_diff",
                 data=to_plot, ax =ax, color = "grey")
    plt.xticks(rotation = 45)  
    
def ensemble(trait = "landcover", folds = 3,
    rf_seeds = np.array(range(5)),
    cv_seeds = np.array(range(5)),
    max_depths = np.array([5,8,10,15,None]),
    buffer_sizes = np.array(range(10000,110000,10000))):
    
    # rf_seeds = [0,1]
    # cv_seeds = [0,1]
    # max_depths = [5, 6]
    # buffer_sizes = [10000,20000]
    
    combos = [rf_seeds, cv_seeds, max_depths, buffer_sizes]
    
    allVars = pd.DataFrame(index = np.array(range(len(rf_seeds)*len(cv_seeds)*len(max_depths)*len(buffer_sizes))), columns = init.trait_keys[trait])
    onlyClimate = allVars.copy()
    
    for sample, (rf_seed, cv_seed, max_depth, buffer_size) in enumerate(itertools.product(*combos)):
        print(f"[INFO] Computing rf_seed:{rf_seed}\tcv_seed:{cv_seed}\tmax_depth:{max_depth}\tbuffer:{buffer_size}")
        
        df, kf, clf = get_combo(rf_seed, cv_seed, buffer_size, max_depth, folds=folds, trait = trait)
        _true_preds = get_true_preds(df, kf, clf, trait = trait)
        allVars.loc[sample] = get_auc_by_trait(_true_preds, trait = trait)
        
        ## without LFMC 
        df =remove_lfmc(df)
        
        _true_preds = get_true_preds(df, kf, clf, trait = trait)
        onlyClimate.loc[sample] = get_auc_by_trait(_true_preds, trait = trait)
    
    diff = allVars - onlyClimate
    return allVars, onlyClimate, diff
    
def main():
    #%% landcover auc plot
    # trait = "landcover"
    # allVars, onlyClimate, diff = ensemble(trait = trait)
    # to_plot = combine_results_into_frame(allVars, onlyClimate, trait = trait)
    # box_plot_lc(to_plot)
    
    #%% p50 auc diff plot
    trait = "p50liu"
    allVars, onlyClimate, diff = ensemble(trait = trait)
    box_plot_trait(diff, trait = trait)
    
    # g=5
if __name__ == "__main__":
    main()
    
