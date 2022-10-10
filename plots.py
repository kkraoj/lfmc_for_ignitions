# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:09:31 2022

@author: kkrao
"""

import os
import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
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
        

def plot_lightnings_ndvi():
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
    raster = r"D:\Krishna\projects\pws_drivers\data\pws_features\usa_dem.tif"
    
    src = rasterio.open(raster,'r')
    left, bottom, right, top = src.bounds
    raster =  rasterio.mask.mask(src, shdf.geometry, crop=False)[0][0]
    # ax.imshow(raster,
    #               cmap="Greys",vmin = 0, vmax = 2000,extent=(left, right, top, bottom), alpha = 0.4
    #               )
    
    
    
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

    # ax.imshow(raster,
    #               cmap="Greys",vmin = 0, vmax = 2000,extent=(left, right, top, bottom), alpha = 0.4
    #               )
    
    ax = plt.figure(figsize=(16,10)).gca(projection=ccrs.PlateCarree())
    projection = ccrs.PlateCarree(central_longitude=0) 
    ax.add_geometries(shdf.geometry,
                      projection,
                      facecolor="None",
                      edgecolor='k', 
                      linewidth = 2)
    
    ax.set_extent([-125,-114,32,43])
    raster = r"D:\Krishna\projects\grid_fire\data\nlcd\nlcd_2016_4km.tif"
    src = rasterio.open(raster,'r')
    left, bottom, right, top = src.bounds

    raster =  rasterio.mask.mask(src, shdf.geometry, crop=False)[0][0]
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
    table = np.flip(np.flip(np.transpose(table), axis = 1), axis = 0)
    
    # print(table)
    # print(np.transpose(table))
    # dfss
    model = statsmodels.stats.contingency_tables.Table2x2(table)

    return [model.riskratio, model.riskratio_confint()[1],  model.riskratio_confint()[0], model.riskratio_pvalue(), 2*len(df)]


def plot_odds_separate_files(ms = 100):
    fs = 14
    mpl.rcParams.update({'font.size': fs,
                         'axes.labelsize':fs,
                         'xtick.labelsize':fs,
                         'ytick.labelsize':fs})
    cols = ["lfmc","fire","lc","mean_ndvi","vpd_4m","agb","mean_t","wind"]    


    or_by_lc = pd.DataFrame(index =["All","Shrub","Forest","Grass"],\
                columns = ["Odds ratio", "upper","lower","p","n"])
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
        
        
    ## applying stars
    or_by_lc['stars'] = 0
    or_by_lc.loc[or_by_lc["p"]<0.05,'stars'] = 1
    or_by_lc.loc[or_by_lc["p"]<0.01,'stars'] = 2
    or_by_lc.loc[or_by_lc["p"]<0.001,'stars'] = 3
    
    or_by_lfmc['stars'] = 0
    or_by_lfmc.loc[or_by_lfmc["p"]<0.05,'stars'] = 1
    or_by_lfmc.loc[or_by_lfmc["p"]<0.01,'stars'] = 2
    or_by_lfmc.loc[or_by_lfmc["p"]<0.001,'stars'] = 3
    
    or_by_year = pd.DataFrame(index =range(2016, 2022), columns = ["Odds ratio", "upper","lower","p","n"])
    all_years = pd.DataFrame()
    for year in range(2016,2022):
        mmdf = pd.DataFrame()
        ssdf = pd.DataFrame()
        for cat in ["forest","shrub","grass"]:
            sdf = pd.read_csv(os.path.join(init.dir_root,\
                                 "data","r","matches",\
                                f"matched_22_apr_2022_extreme_{cat.lower()}_{year}.csv"))
            sdf = sdf.rename(columns = {"y":"fire"})
            mdf = sdf.groupby("match")[cols].apply(reshape)
            mmdf = mmdf.append(mdf, ignore_index = True)
            ssdf = ssdf.append(sdf, ignore_index = True)
        all_years = all_years.append(ssdf, ignore_index = True)
        # pd.to_datetime(ssdf.loc[ssdf["fire"]==1,"date"]).dt.dayofyear.plot(kind = "kde", ax = ax, label = year)
        # (mmdf.lfmc_0- mmdf.lfmc_1).plot(kind = "kde", ax = ax, label = year)
        # print(f"{year} LFMC diff = {(mmdf.loc[mmdf.fire_0==1,'lfmc_0'] - mmdf.loc[mmdf.fire_1==0,'lfmc_1']).mean()}")
        # print(f"{year} LFMC diff = {(mmdf.lfmc_0- mmdf.lfmc_1).mean()}")
        or_by_year.loc[year,:] = get_or_stats(mmdf)
    
    pct_lc_year = all_years.groupby("year").lc.value_counts(normalize = True).round(2)*100
    print(pct_lc_year)
    # ax.set_xlim(-80,0)
    ## applying stars
    or_by_year['stars'] = 0
    or_by_year.loc[or_by_year["p"]<0.05,'stars'] = 1
    or_by_year.loc[or_by_year["p"]<0.01,'stars'] = 2
    or_by_year.loc[or_by_year["p"]<0.001,'stars'] = 3
    
    
    
    fig, axs = plt.subplots(1, 3, figsize =(12,4), sharey = False)

    ax = axs[0] 
    ax.errorbar(or_by_lc.index, or_by_lc["Odds ratio"], \
                yerr = [or_by_lc["Odds ratio"] - or_by_lc["lower"],\
                or_by_lc["upper"] - or_by_lc["Odds ratio"]], \
                    color = "grey", \
                    capsize = 3, zorder = -1, ls='none')
    ax.scatter(or_by_lc.index, or_by_lc["Odds ratio"], \
              color = ['black', 'darkgoldenrod','forestgreen','lawngreen'],\
               edgecolor = "grey", s = ms)
    ax.set_ylim(0,7)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylabel("Risk ratio")
    
    ax = axs[1]

    ax.errorbar(or_by_lfmc.index, or_by_lfmc["Odds ratio"], \
                yerr = [or_by_lfmc["Odds ratio"] - or_by_lfmc["lower"],or_by_lfmc["upper"] - or_by_lfmc["Odds ratio"]], color = "grey", \
                    capsize = 3, zorder = -1)
    ax.scatter(or_by_lfmc.index, or_by_lfmc["Odds ratio"], color = "k", edgecolor = "grey", s = ms)
    ax.set_ylabel("")
    ax.set_xlabel(r"$\Delta$ LFMC in matched pairs (%)")
    new_labs = [f"({int(init.thresh_diff[cat][1])}, {int(init.thresh_diff[cat][0])}]" \
                for cat in init.thresh_diff.keys()]
    # ax.set_xlabel("Range of min. LFMC")
    ax.set_xticklabels(new_labs)
    ax.set_yticks(range(1,5))
    ax.set_ylim(0,4.5)
    
    ax = axs[2]
    
    ax.errorbar(or_by_year.index, or_by_year["Odds ratio"], \
                yerr = [or_by_year["Odds ratio"] - or_by_year["lower"],\
                or_by_year["upper"] - or_by_year["Odds ratio"]], \
                    color = "grey", \
                    capsize = 3, zorder = -1, ls='none')
    ax.scatter(or_by_year.index, or_by_year["Odds ratio"], \
              color = "black",\
               edgecolor = "grey", s = ms)
    ax.set_xticks(range(2016, 2022))
            
    axs[0].annotate("A",xy = (-0.15,1.2), xycoords = "axes fraction")
    axs[0].annotate("Risk ratio binned by land cover",xy = (0.5,1.1), \
                    xycoords = "axes fraction", weight = "bold", ha = "center")
    axs[1].annotate("B",xy = (-0.15,1.2), xycoords = "axes fraction")
    axs[1].annotate(r"Risk ratio binned by $\Delta$ LFMC ",xy = (0.5,1.1), \
                xycoords = "axes fraction", weight = "bold", ha = "center")
    axs[2].annotate("C",xy = (-0.15,1.2), xycoords = "axes fraction")
    axs[2].annotate(r"Risk ratio binned by year",xy = (0.5,1.1), \
                xycoords = "axes fraction", weight = "bold", ha = "center")
    
    for i in range(3):
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].axhline(1, linestyle = "--",color = "grey")

    
    for i, (lc, cat) in enumerate(zip(or_by_lc.index, or_by_lfmc.index)):    
        align = "center"
        axs[0].annotate(f"n = {or_by_lc.loc[lc, 'n']:,}", \
                        xy = (i/3,-0.17), xycoords = "axes fraction", \
                              ha = align)
        axs[0].annotate(f"{'*'*or_by_lc.loc[lc, 'stars']}", \
                        xy = (i,or_by_lc.loc[lc, "upper"]+0.2), \
                              ha = align)
        axs[1].annotate(f"{'*'*or_by_lfmc.loc[cat, 'stars']}", \
                        xy = (i,or_by_lfmc.loc[cat, "upper"]+0.2), \
                              ha = align)
    for i, cat in enumerate(or_by_year.index):
        axs[2].annotate(f"n = {or_by_year.loc[cat, 'n']:,}", \
                        xy = (i/5,-0.35), xycoords = "axes fraction", \
                              rotation = 45, ha = align)
        axs[2].annotate(f"{'*'*or_by_year.loc[cat, 'stars']}", \
                        xy = (cat,or_by_year.loc[cat, "upper"]+0.2), \
                               ha = align)
    ax.set_ylim(0,6)

    
    print(or_by_lc)
    print(or_by_lfmc)        
    print(or_by_year)

def plot_balance(): # do not delete. ms figure made using this
    
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
                     head_width = 0.15, color = "k", linewidth =0.2)
        else:
            ax.arrow(df.Raw[i], i, df.Matched[i]-df.Raw[i]+0.17, 0,
                     head_width = 0.15, color = "k", linewidth =0.2)
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_xlabel("Standardized difference between\ndry-LFMC lightning and wet-LFMC lightning")
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
    # ax.scatter(df.Matched, df.index, marker = "o",s = 40, color = "k")
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

def august_lightnings():
    df = pd.read_csv(os.path.join(init.dir_root, "data","r","rct_22_apr_2022.csv"))

    df = pd.read_csv(os.path.join(init.dir_root, "data",\
            "lightning", "nldn",f"nldn-tiles-15-feb-2022.csv"))
    df.date = pd.to_datetime(df.date)
    df.head()
    df.shape
    df.date.max()
    fausto = df[(df['date']>='2020-08-15') & (df['date']<='2020-08-17')].shape[0]  
    rest = df.shape[0] - fausto
    three_day_periods = ((datetime.date(2021,12,31) - datetime.date(2016,1,1)).days - 3)/3
    rest_avg = rest/three_day_periods
    print(f"Fausto had {fausto} lightning strikes")
    print(f"There were {rest_avg} in other 3-day periods")
    print((fausto - rest_avg)/rest_avg)
    
    
    
    
def main():
    # confusion_matrix()
    # plot_lightnings()
    # plot_lightnings_ndvi()
    plot_odds_separate_files()
    # plot_balance() #Do not delete.  ms figure made using this. 
    # plot_balance_separate_files()
    # plot_gamma()
    # plot_elevation_vs_lc()
    # august_lightnings()
if __name__ == "__main__":
    main()


# fig, ax = plt.subplots()
# a = [0.7,	1.5,	2,	0.3,	4.3,	2.5]
# b = [1, 2, 1.5, 1.75,2.3,1.2]

# ax.plot(a)
# ax2 = ax.twinx()
# ax2.plot(b, color = "red")