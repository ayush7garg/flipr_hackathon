# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:02:12 2020

@author: Ayush Garg
"""

#Importing Libraries
import pandas as pd
import numpy as np
import fbprophet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


#Loading the training dataset
train_data = pd.read_excel('Train_dataset.xlsx').fillna(method='ffill')

#Data Preprocessing
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

#Calculating the correlation between features
cor = train_data.corr()
cor_target = abs(cor["Infect_Prob"])

#Removing the highly correlated independent variables
# Select upper triangle of correlation matrix
upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

x_train = x_train.drop(x_train[to_drop], axis=1)

#Selecting the features having more correlation with the target variable as compared to others
relevant_features = cor_target[cor_target>0.0086]

uncorrelated_features = [feature for feature in train_data if feature not in relevant_features]

x_train = x_train.drop(x_train[uncorrelated_features],axis=1)


#Loading diuresis dataset
xlsx = pd.ExcelFile('Train_dataset.xlsx')
df = pd.read_excel(xlsx, 'Diuresis_TS')
df = df.set_index("people_ID")

df1 = df.T
forecast_values = []
for i in range(len(df1.columns)):    
    y = df1.iloc[:,i]
    y = y.to_frame()
    y.index = y.index.date
    mapping = {y.columns[0]:"y"}
    y = y.rename(columns=mapping)
    y['ds'] = y.index
    y_prophet = fbprophet.Prophet(changepoint_prior_scale=0.8,daily_seasonality=True,n_changepoints=4,yearly_seasonality=False,weekly_seasonality=False)
    y_prophet.fit(y)
    y_forecast = y_prophet.make_future_dataframe(periods=1, freq='D')
    y_forecast = y_prophet.predict(y_forecast)
    f = y_forecast.values[7,15]
    forecast_values.append(f)

diuresis_train = pd.DataFrame((x_train['Diuresis'],forecast_values),columns=['20/03/2020','27/03/2020'])
x_diuresis_train = diuresis_train["20/03/2020"]
y_diuresis_train = diuresis_train["27/03/2020"]

regressor_diuresis = LinearRegression()
regressor_diuresis.fit(x_diuresis_train,y_diuresis_train)

#Loading test dataset
test_data = pd.read_excel('Test_dataset.xlsx')

labelEncoder = LabelEncoder()
test_data['Region'] = labelEncoder.fit_transform(test_data['Region'])
test_data['Gender'] = labelEncoder.fit_transform(test_data['Gender'])
test_data['Occupation'] = labelEncoder.fit_transform(test_data['Occupation'])
test_data['Mode_transport'] = labelEncoder.fit_transform(test_data['Mode_transport'])
test_data['comorbidity'] = labelEncoder.fit_transform(test_data['comorbidity'])
test_data['cardiological pressure'] = labelEncoder.fit_transform(test_data['cardiological pressure'])
test_data['Pulmonary score'] = labelEncoder.fit_transform(test_data['Pulmonary score'])

x_test = test_data.drop(test_data[uncorrelated_features],axis=1)
x_test = x_test.drop(x_test[to_drop],axis=1)

x_diuresis_test = x_test['Diuresis']

y_diuresis_pred = regressor_diuresis.predict(x_diuresis_test)


x_train['Diuresis'] = forecast_values
x_test['Diuresis'] = y_diuresis_pred


#Feature Scaling
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)

#Training the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

x_test = sc_X.fit_transform(x_test)
#Calculating the predictions
y_pred = regressor.predict(x_test)

#Output file
submission = []
submission.append(test_data["people_ID"])
submission.append(y_pred)
submission = np.asarray(submission)
submission = np.transpose(submission)
np.savetxt('Infected Probabilities 27 March 2020.csv',submission,fmt='%d,%f',delimiter=',',header="people_ID,infect_prob")



