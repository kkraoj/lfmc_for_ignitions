# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:22:59 2022

@author: kkrao
"""

import os
import datetime
import re
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics

import init


def make_panel(filepath, state_name, savepath = None, save = True, ):
    df = pd.read_csv(filepath)
    
    df = df.rename(columns = {"vpd_4":"vpd_4m","ppt_1":"ppt_1y","erc_15":"erc_15d"})
    # above line changes col names IF those columns exist. if not, doesn't do anything.
    df.lc = df.lc.round()
    df.fire = 1-df.fire.isnull()
    
    df["longitude"] = df[".geo"].str.split("[").str[1].str.split(",").str[0]
    df["latitude"] = df[".geo"].str.split("[").str[1].str.split(",").str[1].str[:-2]
    
    df.longitude = df.longitude.astype(float).round(1)
    df.latitude = df.latitude.astype(float).round(1)
    
    df["system:time_start"] = df["system:time_start"]/1000/60/60/24
    df["system:time_start"] = df["system:time_start"].astype(int)
    
    
    df = df.dropna() # very important
    df = df.loc[df.lc.isin(init.thresh["extreme"].keys())].copy()
    
    df["z"] = (df.lfmc <= list( map(init.thresh["extreme"].get, df.lc) )).astype(int)
    if save:
        df.to_csv(savepath)
    return df


for state_name in init.states:
    pattern = f"2016_2021_{state_name}"
    all_files = os.listdir(os.path.join(init.dir_root, "data","gee","all_states"))
    filename = [file for file in all_files if pattern in file]
    if len(filename) == 0:
        logging.warning(f"Could not find data for {state_name}. Skipping.")
        continue
    elif len(filename) > 1:
        logging.warning(f"Found multiple files for {state_name}. Skipping.")
        continue
    filename = filename[0]
    filepath = os.path.join(init.dir_root, "data","gee","all_states", filename)
    date = datetime.date.today().strftime("%d-%m-%y")
    savepath = os.path.join(init.dir_root, "data","r",f"rct_{date}_{state_name}.csv")
    
    make_panel(filepath, state_name, savepath)

