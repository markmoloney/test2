t#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers, optimizers, losses
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

np.random.seed(666)


#%% Import data
path = '/Users/alvaro.corrales.canoibm.com/Box/AutoAI Git/01 - Data/Fraud detection'
data = pd.read_csv(path + '/X_train.csv')
target = pd.read_csv(path + '/y_train.csv', header=None)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.1)

#%% 
# ----------------------------------------------------------------------------
# ------------------------------ NEURAL NETWORK ------------------------------
# ----------------------------------------------------------------------------
model = Sequential()
model.add(Dense(X_train.shape[1], activation = 'relu',
                input_dim = X_train.shape[1],
                kernel_regularizer = regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu', 
                kernel_regularizer = regularizers.l2(0.001))) 
model.add(Dense(64, activation = 'relu', 
                kernel_regularizer = regularizers.l2(0.001)))
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(BatchNormalization())   
model.add(Dense(1, activation = 'sigmoid'))

es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=5)
    
model.compile(optimizer = optimizers.Adam(),
              loss = losses.binary_crossentropy,
              metrics=['accuracy'])


model1 = model.fit(X_train, y_train, epochs=200, batch_size = 64,
          validation_split = 0.1,
          callbacks = [es])



#%% Visualise model performance

plt.figure(figsize=(6,4))
plt.plot(model1.history['accuracy'])
plt.plot(model1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()

plt.figure(figsize=(6,4))
plt.plot(model1.history['loss'])
plt.plot(model1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
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
cuisi['real'] = y_train[0]
cuisi['pred_class'] = model.predict_classes(X_train)
cuisi['pred_proba'] = model.predict(X_train)














