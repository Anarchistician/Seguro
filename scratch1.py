#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:29:06 2017
@author: phil


This is partly to refresh my Python and get better with Pandas.
I strongly favor R, so my work here will probably be comparatively sloppy
and excessively commented.

"""


import numpy as np
import pandas as pd

seguro = pd.read_csv('Seguro/train.csv')
seguro.shape # Dimensions
seguro.info()   # Like str in R
seguro.head()

n,m = seguro.shape

### Notes:
##  57 predictors
##  'target' is response
##  'id' is unique identifier

#################################   REMINDER: MISSING VALUES ARE -1

seguro = seguro.replace(-1,np.nan)


##  List number of missing values and percentage of total per column
##  Not particularly efficient, but as long as it runs...
##  I'll have to learn a better way to do this.

nullTest = seguro.loc[:,seguro.isnull().any()]

print( 'Column Name\t Nulls\t Proportion')
for i in range(nullTest.shape[1]):
    print('{}\t{}\t{}'.format(nullTest.columns[i],
          sum(nullTest.isnull().iloc[:,i]),
          round(sum(nullTest.isnull().iloc[:,i])/n,4)))


