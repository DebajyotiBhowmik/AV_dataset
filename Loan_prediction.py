tri#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:43:41 2017

@author: debajyoti
"""

import numpy as np

import pandas as pd

import re
def change_dependent(data):
    ls=[]
    for i in range(data.shape[0]):
        #print(i)
        if type(data[i][0]) is  str:
            data[i][0]=int(re.findall('\d+',data[i][0])[0] )
        else:
            data[i][0]=0

train_data=pd.read_csv("train_loan.csv")

dropped=train_data.iloc[:,1:13]

#dropped=dropped.fillna(dropped.mean())
dropped['Credit_History']=dropped['Credit_History'].fillna(dropped['Credit_History'].mode()[0])
dropped['Loan_Amount_Term']=dropped['Loan_Amount_Term'].fillna(dropped['Loan_Amount_Term'].mode()[0])

dropped=dropped.fillna(dropped.mean())
#dropped=dropped.dropna(axis=0,how='any')
Y_train=dropped.iloc[:,-1].values
dropped=dropped.iloc[:,0:11]
X=pd.get_dummies(dropped,columns=['Gender','Married','Education','Self_Employed','Property_Area'])

X_train=X.iloc[:,0:17].values

change_dependent(X_train)

np.savetxt("clean_train_loan.csv",X_train,comments='',delimiter=',',header="Dependents,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,CreditHistory,Gender_Female,Gender_Male,Married_No,Married_Yes,Education_Graduate,Education_NotGraduate,Self_Employment_No,Self_Employment_Yes,Property_Area_Rural,Property_Area_Semiurban,Property_Area_Urban")

np.savetxt("clean_train_loan_Y.csv",Y_train,fmt="%s",comments='',delimiter=',',header="Loan_Status")
###############fill empty value################
"""from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy="mean",axis=0)

imputer=imputer.fit(X[:,5:10])
X[:,5:10]=imputer.transform(X[:,5:10])"""



from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()

Y_train=labelencoder_X.fit_transform(Y_train)

np.savetxt("clean_train_loan_Y.csv",Y_train,fmt="%s",comments='',delimiter=',',header="Loan_Status")
