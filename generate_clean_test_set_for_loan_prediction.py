#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 22:23:43 2017

@author: debajyoti
"""

import numpy as np

import pandas as pd

import re
def change_dependent(data):
    ls=[]
    for i in range(data.shape[0]):
        #print(i)
        if type(data[i][1]) is  str:
            data[i][1]=int(re.findall('\d+',data[i][1])[0] )
        else:
            data[i][1]=0

test_data=pd.read_csv("test_loan.csv")

#X=test_data.iloc[:,0:12].values
test_data['Gender']=test_data['Gender'].fillna(test_data['Gender'].mode()[0])
test_data['Married']=test_data['Married'].fillna(test_data['Married'].mode()[0])
test_data['Education']=test_data['Education'].fillna(test_data['Education'].mode()[0])
test_data['Property_Area']=test_data['Property_Area'].fillna(test_data['Property_Area'].mode()[0])

test_data['Self_Employed']=test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0])


test_data['Credit_History']=test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0])

X_test=pd.get_dummies(test_data,columns=['Gender','Married','Education','Self_Employed','Property_Area'])

X_test=X_test.iloc[:,0:18].values
change_dependent(X_test)

printable=X_test[:,1:18]
type(X_test[12][1])
np.savetxt("clean_test_loan.csv",printable,comments='',delimiter=',',fmt='%s',header="Dependents,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,CreditHistory,Gender_Female,Gender_Male,Married_No,Married_Yes,Education_Graduate,Education_NotGraduate,Self_Employment_No,Self_Employment_Yes,Property_Area_Rural,Property_Area_Semiurban,Property_Area_Urban")
