# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime

# +
df_tran = pd.read_csv("data/train_transaction.csv", index_col = 'TransactionID')

df_id = pd.read_csv("data/train_identity.csv", index_col = 'TransactionID')
# -

df_tot = df_tran.merge(df_id, how = 'left', left_on='TransactionID', right_on='TransactionID')

df_tot.head()

df_tot.shape


def fill_null(df, attribute_list, stat): 
    for i in attribute_list:     
        print(i)
        print(df[i].dtype)
        if stat == 'median':
            df[i].fillna(df[i].median(), inplace=True)  
        elif stat == 'mean':
            df[i].fillna(df[i].mean(), inplace=True)
        elif stat == 'mode':
            df[i].fillna(df[i].mode()[0], inplace=True)     
        df[i] = df[i].astype(float)
        print(df[i].dtype)
    return df


list(df_tot.isnull())

test = fill_null(df_tot, l, 'mode')

for f in df_tot.columns:
    if df_tot[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_tot[f].values))
        df_tot[f] = lbl.transform(list(df_tot[f].values))

l = ['id_36', 'isFraud']

test = fill_null(df_tot, l, 'mode')


# Extract time stamps
def extract_timestamps(data, START_DATE = '2017-12-01'):
    """
    This function extracts different time stamps from the variable 'TransactionDT'
    such as day of the month, day of the week, hours and minutes and converts them
    into extra variables of the dataframe.
    """
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    data["Date"] = data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    data['Day of the month'] = data['Date'].dt.day
    data['Day of the week'] = data['Date'].dt.dayofweek
    data['Hours'] = data['Date'].dt.hour
    data['Minutes'] = data['Date'].dt.minute
    # data.drop('Date', axis = 1, inplace = True)
    return data


df_tot = extract_timestamps(df_tot)




