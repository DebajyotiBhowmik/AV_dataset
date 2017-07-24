#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:56:32 2017

@author: debajyoti
"""

import numpy as np


import pandas as pd

import matplotlib.pyplot as plt
def generate_defaulter_array(b):
    y=[]
    for i in range(b.shape[0]):
        if b[i]==0:
            y.append('N')
        else:
            y.append('Y')
    return y        
def find_two_class_value_set(X,Y):
    result_class1=[]
    result_class2=[]
    for i in range(X.shape[0]):
        if Y[i]==0:
            result_class1.append(X[i])
        if Y[i]==1:
            result_class2.append(X[i])  
    return tuple(result_class1),tuple(result_class2)        
            
train_data=pd.read_csv('clean_train_loan.csv')

X_train=train_data.iloc[:,0:17].values

Y_train=pd.read_csv('clean_train_loan_Y.csv').iloc[:,0].values


#########################scaling#######################

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
######################################################################
from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=8)

#########################################################################

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,max_iter=100)
########################################################
from sklearn.svm import SVC
classifier=SVC(kernel='linear')

#####################################
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=25,criterion='gini')
####################################################
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
####################
from xgboost import XGBClassifier
classifier=XGBClassifier()
################Test  ####################


X_test=pd.read_csv('clean_test_loan.csv').iloc[:,0:17]
X_test['LoanAmount']=X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean())
X_test['Loan_Amount_Term']=X_test['Loan_Amount_Term'].fillna(X_test['Loan_Amount_Term'].mode()[0])
X_test=X_test.iloc[:,0:17].values
np.any(np.isnan(X_test))
np.any(np.isnan(X_train))

X_test=sc.transform(X_test)


#######################################PCA####################################
from sklearn.decomposition  import PCA
pca=PCA(n_components=12)
X_tr=pca.fit_transform(X_train)
X_te=pca.transform(X_test)

X1,X2=find_two_class_value_set(X_tr,Y_train)
X1=np.array(X1)
X2=np.array(X2)
plt.scatter(X1[:,0],X1[:,1],c='r')
plt.scatter(X2[:,0],X2[:,1],c='b')
explained_variance= pca.explained_variance_ratio_
########################################LDA####################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2)
X_tr=lda.fit_transform(X_train,Y_train)
X_te=lda.transform(X_test)
explained_variance= lda.explained_variance_ratio_
###############################feature selection###############################
from sklearn.feature_selection import RFE
classifier = RFE(classifier, 10)
##############train#############################
classifier.fit(X_train,Y_train)


Y_pred=classifier.predict(X_test)

Y_pred=generate_defaulter_array(Y_pred)

Loan_ID=pd.read_csv('clean_test_loan_Loan_ID.csv').iloc[:,0].values

a=np.array(list(Loan_ID))
a=np.reshape(a,[a.shape[0],1])
b=np.array(list(Y_pred))

b=np.reshape(b,[b.shape[0],1])
result=np.concatenate((a,b),axis=1)

np.savetxt('result.csv',result,delimiter=',',header='Loan_ID,Loan_Status',fmt='%s',comments='')