# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 23:35:23 2022

@author: kkrao
"""


import seaborn as sns
import pandas as pd

df =pd.DataFrame(index = ["wet","dry"],columns =["unburned", "burned"] )
df.index.name = "lfmc"
df.columns.name = "fire"

df.loc[:,:] = [[4838   ,25],[4773,   90]]

df.loc[:,:] = [[666   ,2],[31642,   744]]

sns.heatmap(df, annot=True,cmap = sns.cubehelix_palette(as_cmap=True),\
            fmt="d", cbar=False, linewidths=.8)

# df.loc[:,:] = [[ 7346  ,127],[7256 , 217]]


# sns.heatmap(df, annot=True,cmap = sns.cubehelix_palette(as_cmap=True),\
#             fmt="d", cbar=False, linewidths=.8)


