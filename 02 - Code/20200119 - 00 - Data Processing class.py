#!/usr/bin/env python
# coding: utf-8
# #Please install this package with Python 3.8.1
# pip install -U imbalanced-learn

# +

import numpy as np
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA


# -

# This will update the requirement packages the notebook.
# ! pip freeze > requirements.txt

# Run this to install the packages
# ! pip install -U -r requirements.txt

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


#

class DataProcessing:
    
    def __init__(self, data, target):
        self.data = data.replace({'':np.nan})
        self.target = target
        self.X = self.data.drop(target, axis = 1)
        self.y = self.data[target]
        #self.X = data.loc[:, data.columns != target].values
        #self.y = data.loc[:, [target]].values
    
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
        self.X = self.data.drop(self.target, axis = 1)
        #self.X = self.data.loc[:, self.data.columns != self.target].values
        numeric_columns = []
        for col in self.X.columns:
            if self.X[col].dtype!='object':
                numeric_columns.append(col)
        scaler = preprocessing.StandardScaler().fit(self.X[numeric_columns]) 
        # Now we can standardise
        self.X[numeric_columns] = scaler.transform(self.X[numeric_columns])
        return self.X
    
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

    
    def pca_reduction(self, variance):
        pca = PCA(n_components = variance)
        self.X = pca.fit_transform(self.X)
        return self.X

df_tran = pd.read_csv("../01 - Data/train_transaction.csv", index_col = 'TransactionID')
df_id = pd.read_csv("../01 - Data/train_identity.csv", index_col = 'TransactionID')
df_tot = df_tran.merge(df_id, how = 'left', left_on='TransactionID', right_on='TransactionID')

# +
df = DataProcessing(df_tot, 'isFraud')
df.threshold_col_del(0.25)
df.lblencoder()

attrib_list = list(df.data.columns)
df.fill_null(attrib_list, 'mean', integer = -999)

df.standardiser()
# -

y_train = df.y
X_train = df.X

missing_values_table(X_train)

from sklearn.decomposition import PCA 
pca = PCA(n_components = 0.95)
X2D = pca.fit_transform(X_train)

X2D.shape

X2D.

pca.explained_variance_ratio_

# +
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
colors = ['red', 'green']
plt.scatter(X2D[:, 0], X2D[:, 1], c=y_train, zorder=10, s=2, cmap=matplotlib.colors.ListedColormap(colors))
# -


pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_) 
d = np.argmax(cumsum >= 0.95) + 1

d

pca = PCA(n_components=0.95) 
X_reduced = pca.fit_transform(X_train)







>>>>>>> Stashed changes





