#!/usr/bin/env python
# coding: utf-8
# #Please install this package with Python 3.8.1
# pip install -U imbalanced-learn

# +

import numpy as np
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# -

#

# +

class DataProcessing:
    
    def __init__(self, data, target):
        self.data = data.replace({'':np.nan})
        self.target = target
        self.X = data.loc[:, data.columns != target].values
        self.y = data.loc[:, [target]].values
    
    def threshold_col_del(self, threshold):
        """
        This function keeps only columns that have a share of non-missing values above the threshold. 
        """
        self.data = self.data.dropna(thresh=threshold*len(self.data), axis=1)
        return self.data   
    
    def lblencoder(self):
        """
        This function replaces string variables with encoded values.
        """
        for i in self.data.columns:
            if self.data[i].dtype=='object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.data[i].values))
                self.data[i] = lbl.transform(list(self.data[i].values))
        return self.data
    
    def fill_null(self, attribute_list, stat, integer = -999): 
        """
        This function fills null values of selected columns with one of four different methods:
            - 'median' will fill the nulls with the median of the column. 
            - 'mean' uses the mean of the column. 
            - 'mode' uses the mode of the column. It can be used with string 
            variables, but they need to have been encoded first.
            - 'integer' fills the nulls with an integer (-999 by default).
        """
        for i in attribute_list:     
            #print(self.data[i].dtype)
            if stat == 'median':
                self.data[i].fillna(self.data[i].median(), inplace=True) 
                self.data[i] = self.data[i].astype(float)
            elif stat == 'mean':
                self.data[i].fillna(self.data[i].mean(), inplace=True)
                self.data[i] = self.data[i].astype(float)
            elif stat == 'mode':
                self.data[i].fillna(self.data[i].mode()[0], inplace=True)     
                self.data[i] = self.data[i].astype(int)
            elif stat == 'integer':
                self.data[i].fillna(integer, inplace=True) 
                self.data[i] = self.data[i].astype(float)                
            #print(self.data[i].dtype)
        return self.data
    
    def standardiser(self):
        """
        This function standardises the numeric columns of a dataframe. 
        """
        # Select only numeric features first
        numeric_columns = []
        for col in self.data.columns:
            if self.data[col].dtype!='object':
                numeric_columns.append(col)
        scaler = preprocessing.StandardScaler().fit(self.data[numeric_columns]) 
        # Now we can standardise
        self.data[numeric_columns] = scaler.transform(self.data[numeric_columns])
        return self.data
    def balancesample(self, typ, rs=42):
        self.X = self.data.loc[:, data.columns != target].values
        self.y = self.data.loc[:, [target]].values
        if typ == "under":
            rus = RandomUnderSampler(random_state=rs)
            X_res, y_res = rus.fit_resample(self.X, self.y)
        if typ == "over":
            ros = RandomOverSampler(random_state=rs)
            X_res, y_res = ros.fit_resample(self.X, self.y)
        return X_res, y_res
# -






