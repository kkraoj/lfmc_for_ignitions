# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:51:05 2021

@author: kkrao
"""

import os

import geopandas as gpd
import matplotlib.pyplot as plt

import init

gdf = gpd.read_file(os.path.join(init.dir_root, "data","fire_history","fire20_1.gdb"))
# drop rows years = "" or None
gdf = gdf.loc[gdf.YEAR_.str.len()==4]
gdf.YEAR_ = gdf.YEAR_.astype(int)
n_fires_original = gdf.shape[0]
gdf.YEAR_.unique()
print(f"Fires in FRAP:\t#: {n_fires_original}")
print(f"Fires 2016 onwards:\t#: {gdf.loc[gdf.YEAR_>=2016].shape[0]},\tFraction:{gdf.loc[gdf.YEAR_>=2016].shape[0]/n_fires_original:0.2f}")
print(f"Fires started by lightning:\t#: {gdf.loc[gdf.CAUSE==1].shape[0]},\tFraction:{gdf.loc[gdf.CAUSE==1].shape[0]/n_fires_original:0.2f}")
print(f"Fires 2016 onwards started by lightning:\t#: {gdf.loc[(gdf.CAUSE==1)&(gdf.YEAR_>=2016)].shape[0]},\tFraction:{gdf.loc[(gdf.CAUSE==1)&(gdf.YEAR_>=2016)].shape[0]/n_fires_original:0.2f}")

sub = gdf.loc[(gdf.CAUSE==1)&(gdf.YEAR_==2020)].copy()
sub.GIS_ACRES.hist()

fig, ax = plt.subplots(figsize = (3,3))
ax.hist(sub.GIS_ACRES, bins = 10**np.arange(0,7))
ax.set_xlabel("Fire size (acres)")
ax.set_ylabel("Frequency")
ax.set_xscale('log')





gdf = gpd.read_file(os.path.join(init.dir_root, "data","fire_history","Wildfires_1878_2019_Polygon_Data","Geodatabase","US_Wildfires_1878_2019.gdb"))
gdf.columns
gdf.Combined_Fire_Cause.unique()
causes = ['1 - Lightning','Lightning','lightning', "NATURAL"]
n_fires_original = gdf.shape[0]
print(f"Fires in USGS combined data:\t#: {n_fires_original}")
print(f"Fires 2016 onwards:\t#: {gdf.loc[gdf.Combined_Fire_Year>=2016].shape[0]},\tFraction:{gdf.loc[gdf.Combined_Fire_Year>=2016].shape[0]/n_fires_original:0.2f}")
print(f"Fires started by lightning:\t#: {gdf.loc[gdf.Combined_Fire_Cause.isin(causes)].shape[0]},\tFraction:{gdf.loc[gdf.Combined_Fire_Cause.isin(causes)].shape[0]/n_fires_original:0.2f}")
print(f"Fires 2016 onwards started by lightning:\t#: {gdf.loc[(gdf.Combined_Fire_Cause.isin(causes))&(gdf.Combined_Fire_Year>=2016)].shape[0]},\tFraction:{gdf.loc[(gdf.Combined_Fire_Cause.isin(causes))&(gdf.Combined_Fire_Year>=2016)].shape[0]/n_fires_original:0.2f}")

sub = gdf.loc[(gdf.Combined_Fire_Cause.isin(causes))&(gdf.Combined_Fire_Year>=2016)].copy()
sub.GIS_ACRES.hist()

fig, ax = plt.subplots(figsize = (3,3))
ax.hist(sub.GIS_ACRES, bins = 10**np.arange(0,7))
ax.set_xlabel("Fire size (acres)")
ax.set_ylabel("Frequency")
ax.set_xscale('log')