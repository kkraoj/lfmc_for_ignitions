# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:22:59 2022

@author: kkrao
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics

import init

years = range(2016, 2022)

# df = pd.DataFrame()
# for year in years:
#     df = df.append(pd.read_csv(os.path.join(init.dir_root, "data",\
#             "lightning", "nldn",f"nldn-tiles-{year:4d}.csv"), header=2,\
#             names = ["date","longitude","latitude","count"],
#             dtype = {"date":str,"longitude":float,"latitude":float,"count":int}))
# # df.date = pd.to_datetime(df.date, format = "%Y%m%d")
# df.date = df.date.str[:4]+"-"+df.date.str[4:6]+"-"+df.date.str[6:]
# df.date = df.date.astype(str)
# df.index.name = "fid"
# df.to_csv(os.path.join(init.dir_root, "data",\
#             "lightning", "nldn",f"nldn-tiles-15-feb-2022.csv"))
# # df.head()
# # df.head().to_csv(r"C:\Users\kkrao\Desktop\trial.csv")



# """
# While uploading as asset to GEE leave the Date fiel blank, but in lat and lon
# type in "longitude" and "latitude"
# """


df = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
                               "lightnings_22_feb_2022_2016_2021_California.csv"))
# df = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
                              # "lightnings_11_jan_2022_California.csv"))

    #     ## has lagged vpd 4 months, ppt 1 year, erc 15 days
df = df.rename(columns = {"vpd_4":"vpd_4m","ppt_1":"ppt_1y","erc_15":"erc_15d"})
# df = df.rename(columns = {"vpd":"vpd_4m","ppt":"ppt_1y","erc_15":"erc_15d"})
# rdf = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
#                   "lightnings_18_jan_2022_climatology_ndvi_California.csv"))
# merge_cols = ["fid","system:index","date",".geo"]
# cols = ["mean_t","mean_vpd","mean_ppt","mean_ndvi"]
# df = df.merge(rdf[merge_cols + cols], \
#           on =merge_cols)   
    
# # df = df.join(rdf[])
# rdf = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
#                   "lightnings_18_jan_2022_fire_size_California.csv"))
    
# cols = ["fire_size"]
# df = df.merge(rdf[merge_cols + cols], \
#           on =merge_cols)   

# rdf = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
#               "lightnings_25_jan_2022_prism_instantenous_California.csv"))

# cols = ["vpd","ppt","erc"]
# df = df.merge(rdf[merge_cols + cols], \
#           on =merge_cols) 

# rdf = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
#               "lightnings_25_jan_2022_biomass_canopy_height_California.csv"))
# cols = ["agb","canopy_height"]
# df = df.merge(rdf[merge_cols + cols], \
#           on =merge_cols)
    
# rdf = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
#               "lightnings_31_jan_2022_elevation_California.csv"))
# cols = ["elevation"]
# df = df.merge(rdf[merge_cols + cols], \
#           on =merge_cols)
    
# rdf = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
#               "lightnings_31_jan_2022_ndvi_1m_California.csv"))
# cols = ["ndvi_1m"]
# df = df.merge(rdf[merge_cols + cols], \
#           on =merge_cols)
    
# rdf = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
#               "lightnings_31_jan_2022_wind_1d_California.csv"))
# cols = ["wind"]
# df = df.merge(rdf[merge_cols + cols], \
#           on =merge_cols)


df.lc = df.lc.round()
df.fire = 1-df.fire.isnull()

df["longitude"] = df[".geo"].str.split("[").str[1].str.split(",").str[0]
df["latitude"] = df[".geo"].str.split("[").str[1].str.split(",").str[1].str[:-2]

df.longitude = df.longitude.astype(float).round(1)
df.latitude = df.latitude.astype(float).round(1)

df["system:time_start"] = df["system:time_start"]/1000/60/60/24
df["system:time_start"] = df["system:time_start"].astype(int)


df = df.dropna()
df = df.loc[df.lc.isin(init.thresh["extreme"].keys())].copy()
# df.shape

df["z_extreme"] = (df.lfmc <= list( map(init.thresh["extreme"].get, df.lc) )).astype(int)
df["z_high"] = (df.lfmc <= list( map(init.thresh["high"].get, df.lc) )).astype(int)
df["z_moderate"]= (df.lfmc <= list( map(init.thresh["moderate"].get, df.lc) )).astype(int)
df["z_ndvi"] = (df.ndvi_1m >= df.mean_ndvi).astype(int)
df["z_wind"] = (df.wind >= df.wind.quantile(0.50)).astype(int)
df["z_vpd"] = (df.vpd >= df.mean_vpd).astype(int)
# df.z_moderate.mean()
df.to_csv(os.path.join(init.dir_root, "data","r","rct_22_apr_2022.csv"))
# df.head()['.geo'].apply(get_lon)    
fig, ax = plt.subplots()
sns.kdeplot(x = "lfmc", data =  df.loc[(df.fire==0)],alpha = 0.5,\
            ax = ax, label = "no fire")
sns.kdeplot(x = "lfmc", data =  df.loc[(df.fire==1)],alpha = 0.5,\
            ax = ax, label = "fire")
ax.legend()

"""
    
41.0        5
42.0     9334
43.0       79
52.0    24823
71.0     2886
81.0      177
Name: lc, dtype: int64"""

cm = sklearn.metrics.confusion_matrix(df.fire, df.z_ndvi)
cm[1,1]*cm[0,0]/cm[1,0]/cm[0,1] #= 4.59
print("Extreme\n",sklearn.metrics.confusion_matrix(df.fire, df.z_extreme))
print("High\n",sklearn.metrics.confusion_matrix(df.fire, df.z_high))
print("Moderate\n",sklearn.metrics.confusion_matrix(df.fire, df.z_moderate))

