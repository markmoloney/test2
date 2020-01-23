#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers, optimizers, losses
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from keras.initializers import glorot_normal

seed = 666
np.random.seed(seed)


#%% Import data
path = '/Users/alvaro.corrales.canoibm.com/Box/AutoAI Internal Project/01 - Data/Fraud detection'
train = pd.read_csv(path + '/df_train_split_ppc.csv')
test = pd.read_csv(path + '/df_test_split_ppc.csv')

X_train = train.drop('isFraud', axis = 1).copy()
y_train = train['isFraud'].copy()

X_test = test.drop('isFraud', axis = 1).copy()
y_test = test['isFraud'].copy()

#%% 
# ----------------------------------------------------------------------------
# ------------------------------ NEURAL NETWORK ------------------------------
# ----------------------------------------------------------------------------
model = Sequential()
model.add(Dense(X_train.shape[1], activation = 'relu',
                input_dim = X_train.shape[1],
                kernel_regularizer = regularizers.l2(0.001),
                kernel_initializer = glorot_normal(seed = seed)))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu', 
                kernel_regularizer = regularizers.l2(0.001))) 
model.add(Dense(64, activation = 'relu', 
                kernel_regularizer = regularizers.l2(0.001)))
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu')) 
model.add(Dense(1, activation = 'sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='auto', patience=1)
    
model.compile(optimizer = optimizers.Adam(),
              loss = losses.binary_crossentropy,
              metrics=['accuracy'])


model1 = model.fit(X_train, y_train, epochs=200, batch_size = 64,
          validation_split = 0.1,
          callbacks = [es])



#%% Visualise model performance
# MODEL ACCURACY
plt.figure(figsize=(6,4))
plt.plot(model1.history['accuracy'])
plt.plot(model1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()

# MODEL LOSS
plt.figure(figsize=(6,4))
plt.plot(model1.history['loss'])
plt.plot(model1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()


#%% Test model
y_pred = model.predict_classes(X_test)

print('The ROC-AUC score is:', roc_auc_score(y_test, y_pred))
print('The accuracy score is:', accuracy_score(y_test, y_pred))
print('The precision score is:', precision_score(y_test, y_pred))
print('The recall score is:', recall_score(y_test, y_pred))
print('The F1 score is:', f1_score(y_test, y_pred))










