# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:22:34 2020

@author: saurav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train=pd.read_excel('Train_dataset.xlsx')
df_test=pd.read_excel('Test_dataset.xlsx')
df_test2=pd.read_excel('Test_dataset.xlsx','Put-Call_TS',skiprows=1)

indexes=df_test2.iloc[:,0]

df_test1=df_test2.iloc[:,1:]

df_train.iloc[:,:].isnull().sum()
from sklearn.preprocessing import LabelEncoder

lb1=LabelEncoder()
df_train.iloc[:,1]=lb1.fit_transform(df_train.iloc[:,1])
df_test.iloc[:,1]=lb1.transform(df_test.iloc[:,1])

lb2=LabelEncoder()
df_train.iloc[:,2]=lb2.fit_transform(df_train.iloc[:,2])
df_test.iloc[:,2]=lb2.transform(df_test.iloc[:,2])

#4,9,11

X=df_train.iloc[:,1:14].values
y=df_train.iloc[:,14].values

df_test=df_test.iloc[:,1:].values

from sklearn.preprocessing import Imputer
imp_mode = Imputer(missing_values=np.nan, strategy='most_frequent')
X[:,[3,8,10]]=imp_mode.fit_transform(X[:,[3,8,10]])
df_test[:,[3,8,10]]=imp_mode.transform(df_test[:,[3,8,10]])

imp_mean = Imputer(missing_values=np.nan, strategy='mean')
X[:,[0,1,2,4,5,6,7,9,11,12]]=imp_mean.fit_transform(X[:,[0,1,2,4,5,6,7,9,11,12]])
df_test[:,[0,1,2,4,5,6,7,9,11,12]]=imp_mean.transform(df_test[:,[0,1,2,4,5,6,7,9,11,12]])
df_test1=imp_mean.fit_transform(df_test1)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
mms=MinMaxScaler()
st=StandardScaler()
X[:,2:]=st.fit_transform(X[:,2:])
df_test[:,2:]=st.transform(df_test[:,2:])

X[:,2:]=mms.fit_transform(X[:,2:])
df_test[:,2:]=mms.transform(df_test[:,2:])

#y=st1.fit_transform(y.reshape(len(y),1))



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.25, random_state=0)


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
#from sklearn.svm import SVR

#reg=SVR(C=1000)
reg= RandomForestRegressor(n_estimators=200,criterion='mse')
reg.fit(X_train,y_train)

y_tr_pred=reg.predict(X_train)


y_pred=reg.predict(X_test)

y_pred1=reg.predict(df_test)



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


ret=rmse(y_pred,y_test)
retr=rmse(y_tr_pred,y_train)




#Time Series

from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime

pc_16=[]

for i in range(len(df_test2)):
    #print(i)
    date=pd.date_range(start='08/10/2020', end='08/15/2020')
    date=pd.DataFrame(date)
    pc=df_test1[i,:]
    pc=pd.DataFrame(pc)
    pc=pc.values
    pc=pc.reshape(6,)
    pcd=pd.Series(pc,index=pd.to_datetime(date.iloc[:,0],format='%Y-%m'))
    #pcd_log=np.log(pcd)
    ##pcd_diff=pcd-pcd.shift()
    #pcd_diff.dropna(inplace=True)
    model=ARIMA(pcd,(1,0,0))
    model_fit=model.fit(disp=0)
    
    start_index='2020-08-10'
    end_index='2020-08-16'
    pred=model_fit.predict(start=start_index, end=end_index)
    '''predictions_ARIMA_diff = pd.Series(pred, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(pcd.ix[0])
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,
                                                      fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)'''
    pc_16.append(pred.iloc[6])
    
pc_16=np.asarray(pc_16)
pc_16=pc_16.reshape(len(pc_16),1)
pc_16=mms.fit_transform(pc_16)
pc_16=pc_16.reshape(len(pc_16),)
#pc_16=pd.DataFrame(pc_16)



df_test1=df_test

df_test1[:,11]=pc_16

y_pred2=reg.predict(df_test1)

Submission_part1 = pd.DataFrame({ 'Stock Index': indexes,
                            'Stock Price': y_pred1 })
    
Submission_part2 = pd.DataFrame({ 'Stock Index': indexes,
                            'Stock Price': y_pred2 })
    
Submission_part1.to_csv("Part1.csv",index=False)
Submission_part2.to_csv("Part2.csv",index=False)
    


























