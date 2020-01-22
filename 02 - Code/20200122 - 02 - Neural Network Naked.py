#!/usr/bin/env python
# coding: utf-8

# In[2]:

import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers, optimizers, losses
from keras.callbacks import EarlyStopping
import datetime
import matplotlib.pyplot as plt

np.random.seed(666)


#%% 
# ----------------------------------------------------------------------------
# ------------------------------ NEURAL NETWORK ------------------------------
# ----------------------------------------------------------------------------
model = Sequential()
model.add(Dense(128, activation = 'relu',
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

es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=10)
    
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
cuisi['real'] = y_train
cuisi['pred_class'] = model.predict_classes(X_train)
cuisi['pred_proba'] = model.predict(X_train)














