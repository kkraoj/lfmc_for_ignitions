# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:22:59 2022

@author: kkrao
"""

import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



import init

years = range(2016, 2021)

df = pd.DataFrame()
for year in years:
    df = df.append(pd.read_csv(os.path.join(init.dir_root, "data",\
            "lightning", "nldn",f"nldn-tiles-{year:4d}.csv"), header=2,\
            names = ["date","longitude","latitude","count"],
            dtype = {"date":str,"longitude":float,"latitude":float,"count":int}))
# df.date = pd.to_datetime(df.date, format = "%Y%m%d")
df.date = df.date.str[:4]+"-"+df.date.str[4:6]+"-"+df.date.str[6:]
df.date = df.date.astype(str)
df.index.name = "fid"
df.to_csv(os.path.join(init.dir_root, "data",\
            "lightning", "nldn",f"nldn-tiles.csv"))
df.head()
# df.head().to_csv(r"C:\Users\kkrao\Desktop\trial.csv")
"""
While uploading as asset to GEE leave the Date fiel blank, but in lat and lon
type in "longitude" and "latitude"
"""
