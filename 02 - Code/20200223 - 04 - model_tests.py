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

# Packages 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

# +
# Confusion Matrices & Model Testing
seed = 44
np.random.seed(seed)

df_train = pd.read_csv('../01 - Data/Fraud detection/df_train_split_ppc.csv')
df_test = pd.read_csv('../01 - Data/Fraud detection/df_test_split_ppc.csv')

y_train = df_train['isFraud']
X_train = df_train.drop('isFraud', axis = 1)
y_test = df_test['isFraud']
X_test = df_test.drop('isFraud', axis = 1)

def confusion_matrices(y, y_pred):
    y_pred = y_pred.round()
    confusion_mat = confusion_matrix(y, y_pred)
    sns.set_style("white")
    plt.matshow(confusion_mat, cmap=plt.cm.gray)
    plt.show()
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    normalised_confusion_mat = confusion_mat/row_sums
    print(confusion_mat, "\n")
    print(normalised_confusion_mat)
    plt.matshow(normalised_confusion_mat, cmap=plt.cm.gray)
    plt.show()
    print('the precision score is : ', precision_score(y, y_pred))
    print('the recall score is : ', recall_score(y, y_pred))
    print('the f1 score is : ', f1_score(y, y_pred))
    print('the accuracy score is : ', accuracy_score(y, y_pred))
    print('the accuracy score is : ', roc_auc_score(y, y_pred))
    return

GaussianNB = GaussianNB()
SGDClassifier = SGDClassifier()
RandomForest = RandomForestClassifier(n_estimators=10)
XGBClassifier = XGBClassifier()

scores_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fit_time'])

models = [GaussianNB, SGDClassifier, RandomForest, XGBClassifier]
names = ["Naive Bayes", "SGD Classifier", 'Random Forest Classifier', 'XGB Classifier']

for model, name in zip(models, names):
    temp_list = []
    print(name)

    model.fit(X_train, y_train)
    scores = cross_validate(model, X_train, y_train,
                            scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'),
                            return_train_score=True, cv=10)

    for score in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        mean_score = scores['test_'+score].mean()
        print('{} mean : {}'.format(score, mean_score))
        temp_list.append(mean_score)

    temp_list.append(scores['fit_time'].mean())
    print('average fit time: {}'.format(scores['fit_time'].mean()))
    print("\n")
    scores_df.loc[name] = temp_list
    
scores_df

for i in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fit_time']:
    scores_df.plot.bar(y = i)  
    
test_scores = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

for model, name in zip(models, names):
    temp_list = []
    y_test_pred = model.predict(X_test)

    temp_list.append(accuracy_score(y_test, y_test_pred))
    temp_list.append(precision_score(y_test, y_test_pred))
    temp_list.append(recall_score(y_test, y_test_pred))
    temp_list.append(f1_score(y_test, y_test_pred))
    temp_list.append(roc_auc_score(y_test, y_test_pred))

    test_scores.loc[name] = temp_list

#test_scores

#y_test_pred = GaussianNB.predict(X_test)
#print('Naive Bayes')
#confusion_matrices(y_test, y_test_pred)
#y_test_pred = SGDClassifier.predict(X_test)
#print('SGDClassifier')
#confusion_matrices(y_test, y_test_pred)
#y_test_pred = XGBClassifier.predict(X_test)
#print('XGBClassifier')
#confusion_matrices(y_test, y_test_pred)
#y_test_pred = RandomForest.predict(X_test)
#print('RandomForest')
#confusion_matrices(y_test, y_test_pred)


# +
# Neural Network Setting and Training
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers, optimizers, losses
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal

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

# +
# Visualise Model Performance

# Model Accuracy
plt.figure(figsize=(6,4))
plt.plot(model1.history['accuracy'])
plt.plot(model1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()

# Model Loss
plt.figure(figsize=(6,4))
plt.plot(model1.history['loss'])
plt.plot(model1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()

# Model Performance Stats
#y_pred = model.predict_classes(X_test)
#print('The ROC-AUC score is:', roc_auc_score(y_test, y_pred))
#print('The accuracy score is:', accuracy_score(y_test, y_pred))
#print('The precision score is:', precision_score(y_test, y_pred))
#print('The recall score is:', recall_score(y_test, y_pred))
#print('The F1 score is:', f1_score(y_test, y_pred))

# +
y_pred = model.predict_classes(X_test)

temp_list = []
temp_list.append(accuracy_score(y_test, y_pred))
temp_list.append(precision_score(y_test, y_pred))
temp_list.append(recall_score(y_test, y_pred))
temp_list.append(f1_score(y_test, y_pred))
temp_list.append(roc_auc_score(y_test, y_pred))

test_scores.loc['Neural Network'] = temp_list
# -

test_scores

y_test_pred = model.predict(X_test)
confusion_matrices(y_test, y_test_pred)

y_test_pred = XGBClassifier.predict(X_test)
confusion_matrices(y_test, y_test_pred)

y_test_pred = RandomForest.predict(X_test)
confusion_matrices(y_test, y_test_pred)
