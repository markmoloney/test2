#!/usr/bin/env python
# coding: utf-8
# +
# #Please install this package with Python 3.8.1
# pip install -U imbalanced-learn

import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA

# +
# This will update the requirement packages the notebook.
# #! pip freeze > requirements.txt

# Run this to install the packages
# #! pip install -U -r requirements.txt

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



# -

class DataProcessing:
    
    def __init__(self, data, target):
        self.data = data.replace({'':np.nan})
        #self.data = data
        self.target = target
        self.X = self.data.drop(target, axis = 1)
        self.y = self.data[target]
        
    def threshold_col_del(self, threshold):
        """
        This function keeps only columns that have a share of non-missing values above the threshold. 
        """
        self.data = self.data.dropna(thresh=threshold*len(self.data), axis=1) 
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
    
    def extract_timestamps(self, start_date = '2017-12-01'):
        """
        This function extracts different time stamps from the variable 'TransactionDT'
        such as day of the month, day of the week, hours and minutes and converts them
        into extra variables of the dataframe.
        """
        startdate = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.data["Date"] = self.data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        self.data['Day of the month'] = self.data['Date'].dt.day
        self.data['Day of the week'] = self.data['Date'].dt.dayofweek
        self.data['Hours'] = self.data['Date'].dt.hour
        self.data['Minutes'] = self.data['Date'].dt.minute
        self.data.drop('Date', axis = 1, inplace = True)
        
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
    
    def lblencoder(self):
        """
        This function replaces string variables with encoded values.
        """
        for i in self.data.columns:
            if self.data[i].dtype=='object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.data[i].values))
                self.data[i] = lbl.transform(list(self.data[i].values))
                
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
        
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
        
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
    
    def balancesample(self, typ, rs=42):
        #Updating the self.X and self.y
        self.X = self.data.drop(self.target, axis = 1)
        self.y = self.data[self.target]
        # This conditional statement runs undersampling and oversampling
        # depending on the user's requirements.
        if typ == "under":
            rus = RandomUnderSampler(random_state=rs)
            self.X, self.y = rus.fit_resample(self.X, self.y)
        if typ == "over":
            ros = RandomOverSampler(random_state=rs)
            self.X, self.y = ros.fit_resample(self.X, self.y)

    def standardiser(self):
        """
        This function standardises the numeric columns of a dataframe. 
        """
        # Select only numeric features first

        #self.X = self.data.loc[:, self.data.columns != self.target].values
        numeric_columns = []
        for col in self.X.columns:
            if self.X[col].dtype!='object':
                numeric_columns.append(col)
        scaler = preprocessing.StandardScaler().fit(self.X[numeric_columns]) 
        # Now we can standardise
        self.X[numeric_columns] = scaler.transform(self.X[numeric_columns])       
                      
    def pca_reduction(self, variance):
        pca = PCA(n_components = variance)
        self.X = pca.fit_transform(self.X)
        self.X = pd.DataFrame(self.X)


# df_tran = pd.read_csv("../01 - Data/Fraud detection/train_transaction.csv", index_col = 'TransactionID')
# df_id = pd.read_csv("../01 - Data/Fraud detection/train_identity.csv", index_col = 'TransactionID')
# # Merging the data-set
# df_tot = df_tran.merge(df_id, how = 'left', left_on='TransactionID', right_on='TransactionID')

# + [markdown]
# # # +
# # Testing the Preprocessing class
# df = DataProcessing(df_tot, 'isFraud')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
# df.threshold_col_del(0.25)
# print('threshold')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
# df.extract_timestamps()
# print('timestamps')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
#
# numerical_cols = []
# categorical_cols = []
#
# for col in df.X.columns:
#     if df.X[col].dtype != 'object':
#         numerical_cols.append(col)
#     else:
#         categorical_cols.append(col)
#
# df.lblencoder()
# print('label encoder')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
#
# #attrib_list = list(df.data.columns)
# #df.fill_null(attrib_list, 'mean', integer = -999)
#
# df.fill_null(categorical_cols, 'mode')
# df.fill_null(numerical_cols, 'median')
#
# print('filling in the null values')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
# df.balancesample("over")
# print('balance sample')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
# df.standardiser()
# print('standardiser')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
# df.pca_reduction(0.95)
# print('pca')
# print("the shape of X: ", df.X.shape)
# print("the shape of y: ", df.y.shape,"\n")
# # -
#
# -
df_train_split = pd.read_csv("../01 - Data/df_train_split.csv")
df_test_split = pd.read_csv("../01 - Data/df_test_split.csv")


# +
df_train_split_pp = DataProcessing(df_train_split, 'isFraud')
df_train_split_pp.threshold_col_del(0.25)

df_train_split_pp.extract_timestamps()

numerical_cols = []
categorical_cols = []

for col in df_train_split_pp.X.columns:
    if df_train_split_pp.X[col].dtype != 'object':
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)
df_train_split_pp.lblencoder()
df_train_split_pp.fill_null(categorical_cols, 'mode')
df_train_split_pp.fill_null(numerical_cols, 'median')

df_train_split_pp.balancesample("over")
df_train_split_pp.standardiser()
df_train_split_pp.pca_reduction(0.95)
# -


df_train_split_pp.X.head()
df_train_split_ppc = pd.concat([df_train_split_pp.X, df_train_split_pp.y], axis=1, sort=False)
df_train_split_ppc.to_csv("../01 - Data/df_train_split_ppc.csv", index=False)

# +
df_test_split_pp = DataProcessing(df_test_split, 'isFraud')
df_test_split_pp.threshold_col_del(0.25)
df_test_split_pp.extract_timestamps()

numerical_cols = []
categorical_cols = []

for col in df_test_split_pp.X.columns:
    if df_train_split_pp.X[col].dtype != 'object':
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)
df_test_split_pp.lblencoder()
df_test_split_pp.fill_null(categorical_cols, 'mode')
df_test_split_pp.fill_null(numerical_cols, 'median')
df_test_split_pp.standardiser()

df_test_split_pp.X.head()
df_test_split_ppc = pd.concat([df_test_split_pp.X, df_test_split_pp.y], axis=1, sort=False)
df_test_split_ppc.to_csv("../01 - Data/df_test_split_ppc.csv", index=False)
# -

y_train = df.y
X_train = df.X

df.X.shape

df.y.shape

y_train.value_counts()

X_train.to_csv(r'../01 - Data/02_X_train.csv', index = False)
y_train.to_csv(r'../01 - Data/02_y_train.csv', index = False)


