#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd 
import seaborn as sns
# import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import confusion_matrix



df=pd.read_csv(r"D:\SBME\2nd year\2nd term\biostatistics\diabetes\diabetes.csv")



df.head()



df.info()



df['chol_hdl_ratio'] = df['chol_hdl_ratio'].str.replace(',', '.').astype(float)
df['waist_hip_ratio'] = df['waist_hip_ratio'].str.replace(',', '.').astype(float)
df['bmi'] = df['bmi'].str.replace(',', '.').astype(float)
df



df['gender'].unique()



df['gender']=df['gender'].replace('female',1)
df['gender']=df['gender'].replace('male',0)



df['diabetes'].unique()



df['diabetes']=df['diabetes'].replace('Diabetes',1)
df['diabetes']=df['diabetes'].replace('No diabetes',0)



df.info()


df.drop("patient_number", axis=1, inplace=True)



X = df.drop(['diabetes'] , axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, shuffle =True)



RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=5,random_state=33) 
RandomForestClassifierModel.fit(X_train, y_train)

print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score  is : ' , RandomForestClassifierModel.score(X_test, y_test))



y_pred = RandomForestClassifierModel.predict(X_test)
print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])
y_test[:10]



pickle.dump(RandomForestClassifierModel,open('finalised_model.pkl','wb'))


