# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:18:59 2021

@author: kkrao
"""

dir_data = r"D:\Krishna\projects\lfmc_for_ignitions\data\tables"
dir_root = r"D:\Krishna\projects\lfmc_for_ignitions"
dir_fig = r"D:\Krishna\projects\lfmc_for_ignitions\figures"
lc_dict = { 
            50: 'Closed broadleaf\ndeciduous',
            70: 'Closed needleleaf\nevergreen',
            90: 'Mixed forest',
            100:'Mixed forest',
            110:'Shrub/grassland',
            120:'Shrub/grassland',
            130:'Shrubland',
            140:'Grassland',
            }

short_lc = {'enf':'Closed needleleaf\nevergreen',
            'bdf':'Closed broadleaf\ndeciduous',
            'mixed':'Mixed forest',
            'shrub':'Shrubland',
            'grass': 'Grassland',
            'ENF':'Closed needleleaf\nevergreen',
            'DBF':'Closed broadleaf\ndeciduous',
            'mixed':'Mixed forest',
            'SHB':'Shrubland',
            'GRA': 'Grassland'
            }

color_dict = {'Closed broadleaf\ndeciduous':'darkorange',
              'Closed needleleaf\nevergreen': 'forestgreen',
              'Closed broadleaf deciduous':'darkorange',
              'Closed needleleaf evergreen': 'forestgreen',
              'Mixed forest':'darkslategrey',
              'Shrub/grassland' :'y' ,
              'Shrubland':'tan',
              'Grassland':'lime',
              }  

units = {'lfmc':'(%)','vpd':'(hPa)','erc':'','ppt':r'(mm/month)'}
axis_lims = {'lfmc':[75,125],'vpd':[15,50],'erc':[20,70],'ppt':[0,120]}

lfmc_thresholds = {'Closed broadleaf\ndeciduous':[72,105,125],
              'Closed needleleaf\nevergreen': [72,105,125],
              'Mixed forest':[72,105,125],
              'Shrub/grassland' :[55,67,110] ,
              'Shrubland':[106,121,133],
              'Grassland':[55,67,110],
              } 

trait_keys = {"landcover":sorted(['Closed broadleaf\ndeciduous',
              'Closed needleleaf\nevergreen',
              'Mixed forest',
              'Shrub/grassland',
              'Shrubland',
              'Grassland']),
              "p50":[f"bin {x}" for x in range(10)],
              "sigma":[f"bin {x}" for x in range(10)],
              "p50liu":[f"bin {x}" for x in range(10)],
              "rootdepth":[f"bin {x}" for x in range(10)]
              # "p50":['(-14.100999999999999, -7.2]' , '(-7.2, -5.1]' , '(-5.1, -4.1]' , '(-4.1, -3.9]' , '(-3.9, -1.0]'],
              # "sigma":['(-0.901, 0.4]' , '(0.4, 0.5]' , '(0.5, 0.6]' , '(0.6, 0.7]' , '(0.7, 1.0]'],
              # "p50liu":['(-0.001, 1.6]' , '(1.6, 2.0]' , '(2.0, 4.0]' , '(4.0, 6.0]' , '(6.0, 12.0]'],
              # "rootdepth":['(-0.001, 2.0]' , '(2.0, 3.0]' , '(3.0, 3.7]' , '(3.7, 4.9]' , '(4.9, 17.5]'],
              }