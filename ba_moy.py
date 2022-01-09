# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:53:59 2021

@author: kkrao
"""


import geopandas as gpd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


mpl.rcParams['axes.linewidth'] = 0.5 #set the value globally
mpl.rcParams['ytick.major.width'] = 0.5 #set the value globally
mpl.rcParams['xtick.major.pad'] = 0 #set the value globally

sns.set(style = "white",font_scale= 1)
gdf = gpd.read_file("D:/Krishna/projects/lfmc_for_ignitions/data/fire_history/Wildfires_1878_2019_Polygon_Data/Shapefile/west_1_US_Wildfires_1878_2019.shp")
gdf = gdf.loc[~gdf.IgntDate.isnull()]

gdf.IgntDate = pd.to_datetime(gdf.IgntDate)

# gdf.IgntDate = gdf.IgntDate + pd.DateOffset(months=-2)
gdf['year'] = gdf.IgntDate.dt.year
gdf['month'] = gdf.IgntDate.dt.month


df = gdf.groupby(['year','month']).Hectares.sum().to_frame().reset_index().pivot(index = "year", columns = "month",values = "Hectares")
df = df.loc[df.index>=1972]
df = df.sort_index(ascending = False)
df/=100
df = df.fillna(0)

# make a color map of fixed colors
colors = sns.color_palette('YlOrRd', 8).as_hex() + ["#85132a"]
# colors = colors[::2]
cmap = mpl.colors.ListedColormap(colors)
bounds=[0,1,1e1,50,1e2,500,1e3,5000, 1e4,2e4]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

years = [1972, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2019]
# years = years[::-1]

fig, ax = plt.subplots(figsize = (7,5))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.5, )


sns.heatmap(df, cmap = cmap, ax = ax, norm=norm, cbar_ax = cax, \
            cbar_kws={"ticks":[0,1,10,100,1000,10000,20000],
                    })
cbar = ax.collections[0].colorbar
cbar.ax.set_yticklabels([0,1,10,100,"1,000","10,000","20,000"])
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_yticks([0.5,4,9,14,19,24,29,34,39,44,47])
ax.set_yticklabels(years[::-1])
ax.xaxis.tick_top()
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth('1') 
ax.xaxis.set_ticks_position('none') 
ax.set_xticklabels(["Jan", "Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
cax.set_title("$\quad\quad$Burned area (km$^2$)")
cax.set_frame_on(True)

ax.annotate('Winter$\quad\quad\quad\quad$Spring$\quad\quad\quad\quad$Summer$\quad\quad\quad\quad$Fall$\quad\quad\quad\quad$',xy=(0.4,-0.05),ha = "center", xycoords = "axes fraction")
ax.axvline(2,linewidth = 0.5, color = "k")
ax.axvline(5,linewidth = 0.5, color = "k")
ax.axvline(8,linewidth = 0.5, color = "k")
ax.axvline(11,linewidth = 0.5, color = "k")