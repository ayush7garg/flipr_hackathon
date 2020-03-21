# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:02:12 2020

@author: Ayush Garg
"""


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

xlsx = pd.ExcelFile('Train_dataset.xlsx')
df = pd.read_excel(xlsx, 'Diuresis_TS')

#df = df.set_index('people_ID')
df = df.T

y = df.iloc[0,1:]
df.iloc[59].plot(figsize=(15,6))
plt.show()


decomposition = sm.tsa.seasonal_decompose(df[1], model='additive')
fig = decomposition.plot()
plt.show()