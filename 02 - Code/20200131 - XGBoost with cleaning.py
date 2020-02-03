import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import time 
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#%% Definition of cleaning class
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
        
#%% Data cleaning
path = '/Users/alvaro.corrales.canoibm.com/Box/AutoAI Git/01 - Data/Fraud detection/Original data'
df_tran = pd.read_csv(path + "/train_transaction.csv", index_col = 'TransactionID')
df_id = pd.read_csv(path + "/train_identity.csv", index_col = 'TransactionID')
# Merging the data-set
df_tot = df_tran.merge(df_id, how = 'left', left_on='TransactionID', right_on='TransactionID')



# Set up processing into an object of the newly created class
df = DataProcessing(df_tot, 'isFraud')
print("the shape of X: ", df.X.shape)
print("the shape of y: ", df.y.shape,"\n")

    # Delete columns that don't have at least 25% of valid values
df.threshold_col_del(0.25)
print('threshold')
print("the shape of X: ", df.X.shape)
print("the shape of y: ", df.y.shape,"\n")

    # Extract time stamps (irrelevant imo)
df.extract_timestamps()
print('timestamps')
print("the shape of X: ", df.X.shape)
print("the shape of y: ", df.y.shape,"\n")

    # Identify numerical vs. categorical variables
numerical_cols = []
categorical_cols = []

for col in df.X.columns:
    if df.X[col].dtype != 'object':
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)
        
    # Encode variables
df.lblencoder()
print('label encoder')
print("the shape of X: ", df.X.shape)
print("the shape of y: ", df.y.shape,"\n")

    # Fill nans - All filled with -999
df.fill_null(categorical_cols, 'integer')
df.fill_null(numerical_cols, 'integer')
print('filling in the null values')
print("the shape of X: ", df.X.shape)
print("the shape of y: ", df.y.shape,"\n")
    
    ###################################
    # INCLUDE FEATURE ENGINEERING !!! #
    ###################################

#%%
    # Before continuing, we have to split the dataset into test and split. The
    # transformations that we will apply now would lead to leakage if we didn't 
    # treat test and train sets separately from now on

y = df.y
X = df.X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    # Standardise input - Standardisation based on training set values
numeric_columns = [] # Select only numeric features first
for col in X_train.columns:
    if X_train[col].dtype!='object' and X_test[col].dtype!='object':
        numeric_columns.append(col)
numeric_columns.remove('TransactionDT') #It doesn't make sense to standardise date
        
scaler = preprocessing.StandardScaler().fit(X_train[numeric_columns]) # Now we can standardise
X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])     

    # Oversample
ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(X_train, y_train)

del df_id, df_tran, df_tot, X


#%% MODEL - XGBOOST 
print('Training XGBoost...')
tic = time.time()

clf = xgb.XGBClassifier(
    n_estimators=2000,
    nthread=4,
    max_depth=12,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.4, 
    missing=-999,
    eval_metric='auc',
    tree_method='hist')        

clf.fit(X_train, y_train)

toc = time.time()
print("Time elapsed in fitting XGBoost model: %.2f minutes" % ((toc-tic)/60))

y_pred = clf.predict(X_test)

print('The ROC-AUC score is:', roc_auc_score(y_test, y_pred))
print('The accuracy score is:', accuracy_score(y_test, y_pred))
print('The precision score is:', precision_score(y_test, y_pred))
print('The recall score is:', recall_score(y_test, y_pred))
print('The F1 score is:', f1_score(y_test, y_pred))
