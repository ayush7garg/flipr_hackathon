# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:02:12 2020

@author: Ayush Garg
"""


import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

xlsx = pd.ExcelFile('Train_dataset.xlsx')
df = pd.read_excel(xlsx, 'Diuresis_TS')
df = df.set_index("people_ID")

df1 = df.T

df1.iloc[:,0].plot(figsize=(15, 6))
plt.show()
y = df1.iloc[:,0]
y = y.to_frame()
y.index = df1.index
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = seasonal_decompose(y,freq=1)
fig = decomposition.plot()
plt.show()
