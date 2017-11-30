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
from matplotlib import pyplot as plt
import seaborn as sb
import re
import missingno as mn

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

mn.matrix(seguro)

##  List number of missing values and percentage of total per column
##  Not particularly efficient, but as long as it runs...
##  I'll have to learn a better way to do this.

nullTest = seguro.loc[:,seguro.isnull().any()]
numNulls = [sum(nullTest.isnull().iloc[:,i]) for i in range(nullTest.shape[1])]
propNulls = [ round(i/n,2) for i in numNulls]

print( 'Column Name\t Nulls\t Proportion')
for i in range(len(numNulls)):
    print('{}\t{}\t{}\t'.format(nullTest.columns[i],numNulls[i],propNulls[i]))

##  I'm leaving this to deal with after plots and outliers

"""
Since the data dictionary is sparse,
    I'm going to look at similarly named variables.
"""

seguroNames = list(seguro)
seguroNames
ind = [i for i,s in enumerate(seguroNames) if '_ind_' in s]
reg = [i for i,s in enumerate(seguroNames) if '_reg_' in s]
car = [i for i,s in enumerate(seguroNames) if '_car_' in s]
calc = [i for i,s in enumerate(seguroNames) if '_calc_' in s]

"""

            Ind

"""

indNoBins = [i for i in ind if '_bin' not in seguroNames[i]]

for i in indNoBins:
    plt.figure(i)
    plt.hist(seguro.iloc[:,i].dropna())
    plt.title(seguroNames[i])


indCorr = seguro.iloc[:,indNoBins].dropna().corr()
mask = np.zeros_like(indCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#   Removed bins
#   I can do a pairwise ChiSq analysis on those later.
sb.heatmap(round(indCorr,2),
           mask = mask,
           annot = True,
           vmin = -1,
           vmax = 1,
           cmap = "RdBu")
#Nothing significant.

"""

            Reg

"""

regNoBins = [i for i in reg if '_bin' not in seguroNames[i]]

for i in regNoBins:
    plt.figure(i)
    plt.hist(seguro.iloc[:,i].dropna())
    plt.title(seguroNames[i])


regCorr = seguro.iloc[:,regNoBins].dropna().corr()
mask = np.zeros_like(regCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#   Removed bins
#   I can do a pairwise ChiSq analysis on those later.
sb.heatmap(round(regCorr,2),
           mask = mask,
           annot = True,
           vmin = -1,
           vmax = 1,
           cmap = "RdBu")

##  Correlation between 02 and 03. Let's take a look.
sb.pairplot(seguro.iloc[:,regNoBins].dropna())

"""

            Car

"""

carNoBins = [i for i in car if '_bin' not in seguroNames[i]]

for i in carNoBins:
    plt.figure(i)
    plt.hist(seguro.iloc[:,i].dropna())
    plt.title(seguroNames[i])


carCorr = seguro.iloc[:,carNoBins].dropna().corr()
mask = np.zeros_like(carCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#   Removed bins
#   I can do a pairwise ChiSq analysis on those later.
sb.heatmap(round(carCorr,2),
           mask = mask,
           annot = True,
           vmin = -1,
           vmax = 1,
           cmap = "RdBu")

##  A lot to look at here. Will return.

"""

            Calc

"""

calcNoBins = [i for i in calc if '_bin' not in seguroNames[i]]

for i in calcNoBins:
    plt.figure(i)
    plt.hist(seguro.iloc[:,i].dropna())
    plt.title(seguroNames[i])


calcCorr = seguro.iloc[:,calcNoBins].dropna().corr()
mask = np.zeros_like(calcCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#   Removed bins
#   I can do a pairwise ChiSq analysis on those later.
sb.heatmap(round(calcCorr,2),
           mask = mask,
           annot = True,
           vmin = -1,
           vmax = 1,
           cmap = "RdBu")

##  That ain't right.
calcCorr
round(calcCorr,2)


"""

            Corr Plot Across all Sections

"""

allCorr = seguro.dropna().corr()
mask = np.zeros_like(allCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sb.heatmap(round(allCorr,2),
           mask = mask,
           vmin = -1,
           vmax = 1,
           cmap = "RdBu")

##  Calc looks way too clean to be true.

