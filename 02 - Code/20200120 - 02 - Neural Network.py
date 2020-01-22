#!/usr/bin/env python
# coding: utf-8

# In[2]:

import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers, optimizers, losses
from keras.callbacks import EarlyStopping
import datetime


# In[ ]:

# ----------------------------------------------------------------------------
# ------------------ DELETE AND REPLACE WITH CLEAN DATASET -------------------
# ----------------------------------------------------------------------------

class DataProcessing:
    
    def __init__(self, data):
        self.data = data
    
    def threshold_col_del(self, threshold):
        """
        This function keeps only columns that have a share of non-missing values above the threshold. 
        """
        self.data=self.data.replace({'':np.nan})
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

#%% Imoprt data to test class
tic = time.time()

# Load stuff
path = '/Users/alvaro.corrales.canoibm.com/Box/AutoAI Internal Project/01 - Data/Fraud detection'

train_transaction = pd.read_csv(path + '/train_transaction.csv',
                                index_col='TransactionID')

train_identity = pd.read_csv(path + '/train_identity.csv', 
                             index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, 
                                right_index=True)


# Create a copy of dataframe to compare after all tests
data = train.copy() 

# Extract time stamps
def extract_timestamps(data, START_DATE = '2017-12-01'):
    
    """
    This function extracts different time stamps from the variable 'TransactionDT'
    such as day of the month, day of the week, hours and minutes and converts them
    into extra variables of the dataframe.
    """
    
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    dates = data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    
#    data['Day of the month'] = data['Date'].dt.day
#    data['Day of the week'] = data['Date'].dt.dayofweek
#    data['Hours'] = data['Date'].dt.hour
#    data['Minutes'] = data['Date'].dt.minute
    
#    data.drop('Date', axis = 1, inplace = True)    

    return dates

dates = extract_timestamps(data)

# Split into independent and target variables
data = data.drop('isFraud', axis=1)
target = train['isFraud']

toc = time.time()
print("Time elapsed loading data: %.2f minutes" % ((toc-tic)/60))

#%% Clean data with class
tic = time.time()

data = DataProcessing(data).threshold_col_del(threshold = 0.90)
data = DataProcessing(data).lblencoder()
data = DataProcessing(data).fill_null(data.columns, stat = 'median')
data = DataProcessing(data).standardiser()

data['Dates'] = dates # Incorporate dates
data.sort_values(by = 'Dates', inplace = True) # Sort by date
data.drop('Dates', axis = 1, inplace = True)

## Train-test split based on dates
#X_train = data[:int(0.90*len(data))].copy()
#y_train = target[:int(0.90*len(target))].copy()
#
#X_test = data[int(0.90*len(data)):].copy()
#y_test = target[int(0.90*len(target)):].copy()

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.9, random_state = 666)

toc = time.time()
print("Time elapsed processing data: %.2f minutes" % ((toc-tic)/60))



#%% 
# ----------------------------------------------------------------------------
# ------------------------------ NEURAL NETWORK ------------------------------
# ----------------------------------------------------------------------------
model = Sequential()
model.add(Dense(128, activation = 'relu',
                input_dim = X_train.shape[1],
                kernel_regularizer = regularizers.l2(0.001)))
model.add(Dense(64, activation = 'relu', 
                kernel_regularizer = regularizers.l2(0.001))) 
model.add(Dense(64, activation = 'relu', 
                kernel_regularizer = regularizers.l2(0.001)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))   
model.add(Dense(1, activation = 'sigmoid'))

es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=30)
    
model.compile(optimizer = optimizers.Adam(),
              loss = losses.binary_crossentropy,
              metrics=['accuracy'])


model1 = model.fit(X_train, y_train, epochs=500, batch_size = 64,
          validation_split = 0.1,
          callbacks = [es])



#%% Visualise model performance
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(model1.history['accuracy'])
plt.plot(model1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()


#%% Test model accuracy
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Accuracy in training set:', train_acc)
print('Accuracy in test set:', test_acc)


#%% Shenanigans
cuisi = pd.DataFrame()
cuisi['real'] = y_train
cuisi['pred_class'] = model.predict_classes(X_train)
cuisi['pred_proba'] = model.predict(X_train)














