# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:51 2020

@author: Ayush Garg
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

train_data = pd.read_excel('Train_dataset.xlsx').fillna(method='ffill')

labelEncoder = LabelEncoder()
train_data['Region'] = labelEncoder.fit_transform(train_data['Region'])
train_data['Gender'] = labelEncoder.fit_transform(train_data['Gender'])
train_data['Occupation'] = labelEncoder.fit_transform(train_data['Occupation'])
train_data['Mode_transport'] = labelEncoder.fit_transform(train_data['Mode_transport'])
train_data['comorbidity'] = labelEncoder.fit_transform(train_data['comorbidity'])
train_data['cardiological pressure'] = labelEncoder.fit_transform(train_data['cardiological pressure'])
train_data['Pulmonary score'] = labelEncoder.fit_transform(train_data['Pulmonary score'])


x_train = train_data.drop("Infect_Prob",1)
y_train = train_data["Infect_Prob"]



cor = train_data.corr()
cor_target = abs(cor["Infect_Prob"])

# Select upper triangle of correlation matrix
upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

x_train = x_train.drop(x_train[to_drop], axis=1)


relevant_features = cor_target[cor_target>0.017]

uncorrelated_features = [feature for feature in train_data if feature not in relevant_features]

required_features = x_train.drop(x_train[uncorrelated_features],axis=1)

x_train = required_features

regressor = LinearRegression()
regressor.fit(x_train, y_train)

test_data = pd.read_excel('Test_dataset.xlsx')
test_data['Region'] = labelEncoder.fit_transform(test_data['Region'])
test_data['Gender'] = labelEncoder.fit_transform(test_data['Gender'])
test_data['Occupation'] = labelEncoder.fit_transform(test_data['Occupation'])
test_data['Mode_transport'] = labelEncoder.fit_transform(test_data['Mode_transport'])
test_data['comorbidity'] = labelEncoder.fit_transform(test_data['comorbidity'])
test_data['cardiological pressure'] = labelEncoder.fit_transform(test_data['cardiological pressure'])
test_data['Pulmonary score'] = labelEncoder.fit_transform(test_data['Pulmonary score'])

x_test = test_data.drop(test_data[uncorrelated_features],axis=1)
x_test = x_test.drop(x_test[to_drop],axis=1)
y_pred = regressor.predict(x_test)

submission = []
submission.append(test_data["people_ID"])
submission.append(y_pred)
submission = np.asarray(submission)
submission = np.transpose(submission)
np.savetxt('submission.csv',submission,fmt='%d,%f',delimiter=',',header="people_ID,infect_prob")
