# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:50:32 2022

@author: kkrao
"""

import os

import pandas as pd

import init

csvs = os.listdir(os.path.join(init.dir_root, "data","gee","all_states"))

df = pd.read_csv(os.path.join(init.dir_root, "data","gee",\
                               "lightnings_22_feb_2022_2016_2021_California.csv"))
    
for state in init.states:
    n_state_files = 0
    df = pd.DataFrame()
    for state_csv in csvs:
        filename, _ = os.path.splitext(state_csv)
        filename_state = filename.split("_")[-1]
        if filename_state == state:
            old_filename = state_csv
            n_state_files += 1
            df = df.append(pd.read_csv(os.path.join(init.dir_root, "data","gee",\
                               "all_states", state_csv)), ignore_index = True)
    if n_state_files > 1:
        # need to write new combined file
        new_filename = '_'.join(map(str, old_filename.split("_")[:-2] + [old_filename.split("_")[-1]]))
        df.to_csv(os.path.join(init.dir_root, "data","gee",\
                               "all_states", new_filename), index = False)
