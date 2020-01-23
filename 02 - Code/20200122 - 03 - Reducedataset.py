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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_tran = pd.read_csv("../01 - Data/train_transaction.csv", index_col = 'TransactionID')
df_id = pd.read_csv("../01 - Data/train_identity.csv", index_col = 'TransactionID')
# Merging the data-set
df_tot = df_tran.merge(df_id, how = 'left', left_on='TransactionID', right_on='TransactionID')
df_tran.shape
#df_tran['isFraud'].head()

df_tot_X = df_tot.drop('isFraud', axis = 1)
df_tot_y = df_tot['isFraud']

df_tran_tst = pd.read_csv("../01 - Data/test_transaction.csv", index_col = 'TransactionID')
df_id_tst = pd.read_csv("../01 - Data/test_identity.csv", index_col = 'TransactionID')
df_tran_tst.shape
#df_tran_tst['isFraud'].head()

# Merging the data-set
df_tot_tst = df_tran_tst.merge(df_id_tst, how = 'left', left_on='TransactionID', right_on='TransactionID')
df_tot_tst.head()

X_res, X_train_new, y_res, y_train_new = train_test_split(df_tot_X, df_tot_y, test_size=0.12, random_state=42, stratify = df_tot_y)

# X_res_tst, X_test_new, y_res_tst, y_test_new = train_test_split(df_tot_X_tst, df_tot_y_tst, test_size=0.12, random_state=42, stratify = True)

df_train_new = pd.concat([X_train_new, y_train_new], axis=1, sort=False)

df_train_new.to_csv("../01 - Data/new_train.csv", index=False)

# df_test_new = X_test_new.merge(y_test_new)
