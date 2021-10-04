# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:05:28 2021

@author: kkrao
"""
import os

import gdal
import rasterio

import init

ds = gdal.Open(os.path.join(init.dir_root, "data","lightning","OR_GLM-L0_G17_s20182120600520_e20182120605519_c20182120605521.nc"))


for key, value in ds.GetMetadata().items():
    print("{:35}: {}".format(key, value))
    
src = rasterio.open(os.path.join(init.dir_root, "data","lightning","OR_GLM-L0_G17_s20182120605520_e20182120610520_c20182120610522.nc"))

    

array = src.read(3)
