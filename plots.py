# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:09:31 2022

@author: kkrao
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors 

import seaborn as sns
import init

import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.feature import ShapelyFeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd

import rasterio
import rasterio.mask
import fiona
import sklearn.metrics
import statsmodels.stats.contingency_tables
import matplotlib.ticker as ticker


sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")

WIDTH = 3



def plot_lightnings():
    df = pd.DataFrame()
    for cat in ['grass','shrub','forest']:
        sdf = pd.read_csv(os.path.join(init.dir_root,\
                             "data","r","matches",\
                            f"matched_22_apr_2022_extreme_{cat.lower()}.csv"))
        df = df.append(sdf, ignore_index = True)
        
    df = df.rename(columns = {"y":"fire"})
    
    
    fname = r"D:\Krishna\projects\wildfire_from_lfmc\data\CA_State_TIGER2016\CA_State_TIGER2016.shp"
    
    
    shdf = gpd.read_file(fname)
    shdf = shdf.to_crs("EPSG:4326")
    res = "high"
    ax = plt.figure(figsize=(16,10)).gca(projection=ccrs.PlateCarree())
    projection = ccrs.PlateCarree(central_longitude=0) 
    ax.add_geometries(shdf.geometry,
                      projection,
                      facecolor="None",
                      edgecolor='k', 
                      linewidth = 2)
    
    ax.set_extent([-125,-114,32,43])
    ax.scatter(df.loc[df.fire==0,'longitude'], df.loc[df.fire==0,'latitude'], s = 20, color = "blueviolet", \
               alpha = 1, edgecolor = "grey")
    ax.scatter(df.loc[df.fire==1,'longitude'], df.loc[df.fire==1,'latitude'], s = 20, color = "yellow", \
               alpha = 1, edgecolor = "grey")
    raster = r"D:\Krishna\projects\grid_fire\data\nlcd\nlcd_2016_4km.tif"
    
    src = rasterio.open(raster,'r')
    left, bottom, right, top = src.bounds
    raster =  rasterio.mask.mask(src, shdf.geometry, crop=False)[0][0]
    np.unique(raster)
    raster = np.ma.masked_where(\
                ~np.isin(raster, list(init.thresh["extreme"].keys())), \
                raster)
    cmap, norm = matplotlib.colors.from_levels_and_colors([0,50,55,100],['darkgreen','darkgoldenrod','lime'])
    ax.imshow(raster,
                  cmap=cmap, norm = norm,extent=(left, right, top, bottom), alpha = 0.4
                  )
    
    
    
    ax = plt.figure(figsize=(16,10)).gca(projection=ccrs.PlateCarree())
    projection = ccrs.PlateCarree(central_longitude=0) 
    ax.add_geometries(shdf.geometry,
                      projection,
                      facecolor="None",
                      edgecolor='k', 
                      linewidth = 2)
    
    ax.set_extent([-125,-114,32,43])
    ax.scatter(df.loc[:,'longitude'], df.loc[:,'latitude'], s = 20, c = df["z_extreme"],cmap = matplotlib.colors.ListedColormap(["aqua","peru"]), \
               alpha = 1, edgecolor = "grey")
    # ax.scatter(df.loc[df.z_extreme==1,'longitude'], df.loc[df.z_extreme==1,'latitude'], s = 20, color = "peru", \
               # alpha = 1, edgecolor = "grey")
    raster = r"D:\Krishna\projects\grid_fire\data\nlcd\nlcd_2016_4km.tif"
    
    src = rasterio.open(raster,'r')
    left, bottom, right, top = src.bounds
    raster =  rasterio.mask.mask(src, shdf.geometry, crop=False)[0][0]
    np.unique(raster)
    raster = np.ma.masked_where(\
                ~np.isin(raster, list(init.thresh["extreme"].keys())), \
                raster)
    cmap, norm = matplotlib.colors.from_levels_and_colors([0,50,55,100],['darkgreen','darkgoldenrod','lime'])
    ax.imshow(raster,
                  cmap=cmap, norm = norm,extent=(left, right, top, bottom), alpha = 0.4
                  )
        
        
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
 
def get_or_stats(df):
    
    table = sklearn.metrics.confusion_matrix(\
                     np.append(df.fire_0.values,df.fire_1.values), \
                     np.append(np.repeat(1,len(df)),np.repeat(0,len(df))))
    model = statsmodels.stats.contingency_tables.Table2x2(table)

    return [model.oddsratio, model.oddsratio_confint()[1],  model.oddsratio_confint()[0], model.oddsratio_pvalue(), 2*len(df)]

               
def plot_odds(ms = 100):
    df = pd.read_csv(os.path.join(init.dir_root,\
                                 "data","r",\
                                "matched_extreme_21_apr_2022.csv"))
    df = df.rename(columns = {"y":"fire"})
    
    cols = ["lfmc","fire","lc","mean_ndvi","vpd_4m","agb","mean_t","wind"]    
    mdf = df.groupby("match")[cols].apply(reshape)
    mdf["lfmc_diff"] = mdf["lfmc_0"] - mdf["lfmc_1"]
    mdf.dropna(inplace = True)

    mdf["thresh"] = list(map(init.thresh["extreme"].get, mdf.lc_0))

    fig = plt.figure(constrained_layout=True, figsize =(5.5,3.5))
    widths = [4,3]
    heights = [1,1]
    
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                          )
    axs = []
    axs.append(fig.add_subplot(spec[0, 0]))
    axs.append(fig.add_subplot(spec[0, 1], sharey = axs[0]))
        
    ax = axs[1]    

    or_by_lfmc = pd.DataFrame(index = init.thresh_diff.keys(), columns = ["Odds ratio", "upper","lower","p","n"])
    
    for cat in init.thresh_diff.keys():
        sdf = mdf.loc[(mdf.lfmc_diff > init.thresh_diff[cat][0])&(mdf.lfmc_diff <= init.thresh_diff[cat][1])].copy()
        print(sdf.shape)
        or_by_lfmc.loc[cat,:] = get_or_stats(sdf)
        
    ax.errorbar(or_by_lfmc.index, or_by_lfmc["Odds ratio"], \
                yerr = [or_by_lfmc["Odds ratio"] - or_by_lfmc["lower"],or_by_lfmc["upper"] - or_by_lfmc["Odds ratio"]], color = "grey", \
                    capsize = 3, zorder = -1)
    ax.scatter(or_by_lfmc.index, or_by_lfmc["Odds ratio"], color = "k", edgecolor = "grey", s = ms)
    ax.axhline(1, linestyle = "--",color = "grey")
    ax.set_ylabel("")
    ax.set_xlabel(r"$\Delta$ LFMC in matched pair")
    new_labs = [f"({init.thresh_diff[cat][1]}, {init.thresh_diff[cat][0]}]" for cat in init.thresh_diff.keys()]
    # ax.set_xlabel("Range of min. LFMC")
    ax.set_xticklabels(new_labs)

         
    ax = axs[0]    
     
    mdf = mdf.replace({"lc_0": init.lc_dict})
    or_by_lc = pd.DataFrame(index =["All"] + list(mdf.lc_0.unique()) , columns = ["Odds ratio", "upper","lower","p","n"])
    
    for cat in or_by_lc.index:
        if cat == "All":
            sdf = mdf.copy()
        else:
            sdf = mdf.loc[mdf.lc_0==cat].copy()
        or_by_lc.loc[cat,:] = get_or_stats(sdf)
    ax = axs[0]
    
    ax.errorbar(or_by_lc.index, or_by_lc["Odds ratio"], \
                yerr = [or_by_lc["Odds ratio"] - or_by_lc["lower"],\
                or_by_lc["upper"] - or_by_lc["Odds ratio"]], \
                    color = "grey", \
                    capsize = 3, zorder = -1, ls='none')
    ax.scatter(or_by_lc.index, or_by_lc["Odds ratio"], \
              color = ['black', 'forestgreen','darkgoldenrod','lawngreen'],\
               edgecolor = "grey", s = ms)
    ax.set_ylim(0,12)
    ax.axhline(1, linestyle = "--",color = "grey")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylabel("Odds ratio")
    axs[0].annotate("A",xy = (-0.25,1.2), xycoords = "axes fraction")
    axs[0].annotate("Odds ratio per land cover",xy = (0.5,1.1), \
                    xycoords = "axes fraction", weight = "bold", ha = "center")
    axs[0].annotate("B",xy = (1,1.2), xycoords = "axes fraction")
    axs[1].annotate(r"Odds ratio binned by $\Delta$ LFMC ",xy = (0.5,1.1), \
                xycoords = "axes fraction", weight = "bold", ha = "center")
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

def plot_odds_separate_files(ms = 100):

    cols = ["lfmc","fire","lc","mean_ndvi","vpd_4m","agb","mean_t","wind"]    


    or_by_lc = pd.DataFrame(index =["All","Shrub","Forest","Grass"] , columns = ["Odds ratio", "upper","lower","p","n"])
    mmdf = pd.DataFrame()
    for cat in or_by_lc.index:
        if cat != "All":
            sdf = pd.read_csv(os.path.join(init.dir_root,\
                                 "data","r","matches",\
                                f"matched_22_apr_2022_extreme_{cat.lower()}.csv"))
            sdf = sdf.rename(columns = {"y":"fire"})
            mdf = sdf.groupby("match")[cols].apply(reshape)    
            or_by_lc.loc[cat,:] = get_or_stats(mdf)
            mmdf = mmdf.append(mdf, ignore_index = True)
    or_by_lc.loc["All",:] = get_or_stats(mmdf)
    mmdf["lfmc_diff"] = mmdf["lfmc_0"] - mmdf["lfmc_1"]
    or_by_lfmc = pd.DataFrame(index = init.thresh_diff.keys(), columns = ["Odds ratio", "upper","lower","p","n"])
    for cat in init.thresh_diff.keys():
        mdf = mmdf.loc[(mmdf.lfmc_diff > init.thresh_diff[cat][0])&\
                       (mmdf.lfmc_diff <= init.thresh_diff[cat][1])].copy()
        or_by_lfmc.loc[cat,:] = get_or_stats(mdf)
        
    fig, axs = plt.subplots(1, 2, figsize =(8,4), sharey = True)

    ax = axs[0] 
    ax.errorbar(or_by_lc.index, or_by_lc["Odds ratio"], \
                yerr = [or_by_lc["Odds ratio"] - or_by_lc["lower"],\
                or_by_lc["upper"] - or_by_lc["Odds ratio"]], \
                    color = "grey", \
                    capsize = 3, zorder = -1, ls='none')
    ax.scatter(or_by_lc.index, or_by_lc["Odds ratio"], \
              color = ['black', 'darkgoldenrod','forestgreen','lawngreen'],\
               edgecolor = "grey", s = ms)
    ax.set_ylim(0,8)
    ax.axhline(1, linestyle = "--",color = "grey")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylabel("Odds ratio")
    
    ax = axs[1]

    ax.errorbar(or_by_lfmc.index, or_by_lfmc["Odds ratio"], \
                yerr = [or_by_lfmc["Odds ratio"] - or_by_lfmc["lower"],or_by_lfmc["upper"] - or_by_lfmc["Odds ratio"]], color = "grey", \
                    capsize = 3, zorder = -1)
    ax.scatter(or_by_lfmc.index, or_by_lfmc["Odds ratio"], color = "k", edgecolor = "grey", s = ms)
    ax.axhline(1, linestyle = "--",color = "grey")
    ax.set_ylabel("")
    ax.set_xlabel(r"$\Delta$ LFMC in matched pairs")
    new_labs = [f"({int(init.thresh_diff[cat][1])}, {int(init.thresh_diff[cat][0])}]" \
                for cat in init.thresh_diff.keys()]
    # ax.set_xlabel("Range of min. LFMC")
    ax.set_xticklabels(new_labs)
    
    
    
    
    axs[0].annotate("A",xy = (-0.25,1.2), xycoords = "axes fraction")
    axs[0].annotate("Odds ratio per land cover",xy = (0.5,1.1), \
                    xycoords = "axes fraction", weight = "bold", ha = "center")
    axs[0].annotate("B",xy = (1,1.2), xycoords = "axes fraction")
    axs[1].annotate(r"Odds ratio binned by $\Delta$ LFMC ",xy = (0.5,1.1), \
                xycoords = "axes fraction", weight = "bold", ha = "center")
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    
    for i, (lc, cat) in enumerate(zip(or_by_lc.index, or_by_lfmc.index)):    
        delta = 0
        align = "center"
        if i==0:
            align = "left"
            delta = -0.1
        elif i ==3:
            align = "right"
            delta = 0.1

        axs[0].annotate(f"n = {or_by_lc.loc[lc, 'n']:,}", \
                        xy = (i+delta,-0.1), \
                              ha = align)
        # axs[1].annotate(f"n = {or_by_lfmc.loc[cat, 'n']:,}", \
                        # xy = (i+delta,or_by_lfmc.loc[cat, "upper"]+0.3), \
                              # ha = align)
    print(or_by_lc)
    print(or_by_lfmc)        

def plot_balance():
    
    df = pd.read_csv(os.path.join(init.dir_root, "data","r","balance_16_mar_2022.csv"), index_col = 0)
    df.index = df.index.map(init.var_names)
    df = df.rename(columns = {"results.std.diff.Unadj":"Raw","results.std.diff.ms.1":"Matched"})
    print(df["Matched"].abs().mean())
    fig, ax = plt.subplots(figsize = (3,3.5))
    ax.scatter(df.Raw, df.index, marker = "s",s = 40, color = "k")
    ax.scatter(df.Matched, df.index, marker = "o",s = 40, color = "k")
    ax.axvline(0, color = "k")
    for i in range(len(df.index)):
        if df.Matched[i]-df.Raw[i]>0:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]-0.2, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
        else:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]+0.17, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
    ax.yaxis.set_ticks_position('none') 

    ax.set_xlabel("Standardized difference")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plot_balance_separate_files():
    lcs = ["shrub","forest","grass"]

    df = pd.DataFrame()
    for lc in lcs:
        sdf = pd.read_csv(os.path.join(init.dir_root, "data","r","balance",\
                                       f"balance_24_apr_2022_{lc}.csv"), index_col = 0)
        sdf.index = sdf.index.map(init.var_names)
        sdf = sdf.rename(columns = {"results.std.diff.Unadj":"Raw","results.std.diff.ms.1":"Matched"})
        sdf= sdf[["Raw","Matched"]]
        sdf["n"] = pd.read_csv(os.path.join(init.dir_root,\
                             "data","r","matches",\
                            f"matched_22_apr_2022_extreme_{lc}.csv")).shape[0]
        sdf.columns = [col + f"_{lc}" for col in sdf.columns]
        df = df.join(sdf, how = "outer")
    
    for col in ["Raw","Matched"]:
        df[col] = (df[f"{col}_shrub"]*df["n_shrub"]+\
                    df[f"{col}_forest"]*df["n_forest"]+\
                    df[f"{col}_grass"]*df["n_grass"])/\
                        (df["n_shrub"]+df["n_forest"]+df["n_grass"])
    print(df["Matched"].abs().mean())
    fig, ax = plt.subplots(figsize = (3,3.5))
    ax.scatter(df.Raw, df.index, marker = "s",s = 40, color = "k")
    ax.scatter(df.Matched, df.index, marker = "o",s = 40, color = "k")
    ax.axvline(0, color = "k")
    for i in range(len(df.index)):
        if df.Matched[i]-df.Raw[i]>0:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]-0.2, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
        else:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]+0.17, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
    ax.yaxis.set_ticks_position('none') 

    ax.set_xlabel("Standardized difference")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-1.5,1.5)
    
def plot_gamma():
    fig, ax = plt.subplots(figsize = (3,3))
    for lc in ["all"]:
        sdf = pd.read_csv(os.path.join(init.dir_root,\
                                 "data","r","pvalues",\
                                f"p_gamma_24_apr_2022_{lc}.csv"), usecols = [1,2],index_col = 0)
        sdf.plot(ax = ax, color = init.lc_color[lc], linewidth = 1, legend = False)
    
    x = sdf.index[(sdf['pvalue']-0.05).abs().argmin()]
    ax.hlines(0.05,1,x, color = "grey")
    ax.vlines(x,0,0.05, color = "grey")
    ax.set_ylabel("P value")
    ax.set_xlabel("Confounding ratio")
    ax.set_xlim(1,1.6)
    ax.set_ylim(0,0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.annotate(f"{x:0.2f}", xy = (x, 0.005))


def confusion_matrix():
    
    df = pd.read_csv(os.path.join(os.path.join(init.dir_root, "data","r","rct_22_apr_2022.csv")))
    df = sklearn.metrics.confusion_matrix(df.fire, df.z_extreme)
    print(df)
    print(sum(sum(df)))
    # sdf = df.loc[df.lc==71]
    # print(sklearn.metrics.confusion_matrix(sdf.fire, sdf.z_high))
    # df = pd.read_csv(os.path.join(init.dir_root,\
                                 # "data","r",\
                                # "matched_extreme_3_mar_2022.csv"))
    df = np.zeros((2,2))
                                
    for lc in ["Shrub","Forest","Grass"]:
        sdf = pd.read_csv(os.path.join(init.dir_root,\
                                 "data","r","matches",\
                                f"matched_22_apr_2022_extreme_{lc.lower()}.csv"))
        sdf = sklearn.metrics.confusion_matrix(sdf.y, sdf.z_extreme)
        print(lc)
        print(sdf)
        df += sdf
    print(df)
    print(sum(sum(df)))

def plot_my_balance():
    df = pd.read_csv(os.path.join(os.path.join(init.dir_root, "data","r","rct_22_apr_2022.csv")))
    rows = ['longitude','latitude','agb','wind','vpd_4m','ppt_1y']
    
    balance_raw = pd.pivot_table(df, values = rows, columns = ["z_extreme"])
    std = df[rows].std()
    balance_raw = (balance_raw[1] - balance_raw[0])/std
    
    
    df = pd.read_csv(os.path.join(init.dir_root,\
                                 "data","r",\
                                "matched_extreme_21_apr_2022.csv"))
    balance_matched = pd.pivot_table(df, values = rows, columns = ["z_extreme"])
    balance_matched = (balance_matched[1] - balance_matched[0])/std
    
    df = pd.DataFrame({"Raw":balance_raw, "Matched":balance_matched})
    # df = df.rename(index = {'longitude':'Longitude', 'latitude':'Latitude', 'vpd_4m':'VPD$_{\rm 4\ months\ mean}$',\
    #    'ppt_1y':'P$_{\rm 12\ months\ sum}$', 'agb':'AGB', 'wind':'Wind speed'})
    
    fig, ax = plt.subplots(figsize = (3,3.5))
    ax.scatter(df.Raw, df.index, marker = "s",s = 40, color = "k")
    ax.scatter(df.Matched, df.index, marker = "o",s = 40, color = "k")
    ax.axvline(0, color = "k")
    for i in range(len(df.index)):
        if df.Matched[i]-df.Raw[i]>0:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]-0.2, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
        else:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]+0.17, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
    ax.yaxis.set_ticks_position('none') 

    ax.set_xlabel("Standardized difference")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
    
    
    
    df = pd.DataFrame()
    for cat in ['grass','shrub','forest']:
        sdf = pd.read_csv(os.path.join(init.dir_root,\
                             "data","r","matches",\
                            f"matched_22_apr_2022_extreme_{cat.lower()}.csv"))
        df = df.append(sdf, ignore_index = True)
    balance_matched = pd.pivot_table(df, values = rows, columns = ["z_extreme"])
    balance_matched = (balance_matched[1] - balance_matched[0])/std
    
    df = pd.DataFrame({"Raw":balance_raw, "Matched":balance_matched})
    # df = df.rename(index = {'longitude':'Longitude', 'latitude':'Latitude', 'vpd_4m':'VPD$_{\rm 4\ months\ mean}$',\
    #    'ppt_1y':'P$_{\rm 12\ months\ sum}$', 'agb':'AGB', 'wind':'Wind speed'})
    
    fig, ax = plt.subplots(figsize = (3,3.5))
    ax.scatter(df.Raw, df.index, marker = "s",s = 40, color = "k")
    ax.scatter(df.Matched, df.index, marker = "o",s = 40, color = "k")
    ax.axvline(0, color = "k")
    for i in range(len(df.index)):
        if df.Matched[i]-df.Raw[i]>0:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]-0.2, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
        else:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]+0.17, 0,
                     head_width = 0.1, color = "k", linewidth =0.2)
    ax.yaxis.set_ticks_position('none') 

    ax.set_xlabel("Standardized difference")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    

    
    
def plot_elevation_vs_lc():
    df = pd.DataFrame()
    for lc in ["Shrub","Forest","Grass"]:
        sdf = pd.read_csv(os.path.join(init.dir_root,\
                                 "data","r","matches",\
                                f"matched_22_apr_2022_extreme_{lc.lower()}.csv"))
        df = df.append(sdf, ignore_index = True)
        
    df.lc = df.lc.map(init.lc_dict)
    fig, ax = plt.subplots(figsize = (4,4))
    sns.boxplot(x="lc", y="elevation", data=df, ax= ax, \
               palette = ['darkgoldenrod','forestgreen','lawngreen'])
    ax.set_xlabel("")
    ax.set_ylabel("Elevation (m)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
def main():
    # confusion_matrix()
    # plot_lightnings()
    # plot_odds()
    plot_odds_separate_files()
    # plot_balance()
    # plot_balance_separate_files()
    # plot_gamma()
    # plot_elevation_vs_lc()
if __name__ == "__main__":
    main()
