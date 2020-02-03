import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import time 
import xgboost as xgb
import pickle

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
        
#%% Read in data
path = '/Users/alvaro.corrales.canoibm.com/Desktop/Scripts/AutoAI internal project/1 - Data'

    # Training dataset 
train_tran = pd.read_csv(path + "/train_transaction.csv", index_col = 'TransactionID')
train_id = pd.read_csv(path + "/train_identity.csv", index_col = 'TransactionID')
    # Merging training dataset
train_tot = train_tran.merge(train_id, how = 'left', left_on='TransactionID', right_on='TransactionID')
train_tot['Test'] = 0 # Train/test set identifier

    # Testing dataset
test_tran = pd.read_csv(path + "/test_transaction.csv", index_col = 'TransactionID')
test_id = pd.read_csv(path + "/test_identity.csv", index_col = 'TransactionID')
    # Merging testing dataset
test_tot = test_tran.merge(test_id, how = 'left', left_on='TransactionID', right_on='TransactionID')
test_tot['Test'] = 1 # Train/test set identifier


    # Append test set to training set for cleaning 
    # Names are not exactly equal in test_id and train_id, so will have to fix it

    # Get lists of names to be changed in both training and test sets
id_cols_test = list(test_id.columns)   
id_cols_train = list(train_id.columns)
for list_names in id_cols_train, id_cols_test:
    list_names.remove('DeviceType')
    list_names.remove('DeviceInfo')

    # Get a list of new names and map it to dictionaries
new_names = ['id_' + str(i) for i in range(1, 1 + len(id_cols_test))]
new_cols_train = dict(zip(id_cols_train, new_names))
new_cols_test = dict(zip(id_cols_test, new_names))

    # Now we rename consistently across training and test set
test_tot.rename(columns = new_cols_test, inplace = True)
train_tot.rename(columns = new_cols_train, inplace = True)

    # After renaming, we can append both datasets
df_tot = train_tot.append(test_tot, sort = False)

    # Get rid of stuff that we don't need anymore
del new_names, new_cols_train, new_cols_test, train_tot, test_tot, list_names
del test_tran, test_id, train_id, train_tran, id_cols_test, id_cols_train

#%% Clean data

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
    
    # 1 - NXOR on geographical variables - Done
    # 2 - Round/Truncate amounts - Done
    # 3 - Multiple of 5 
    # 4 - Decimal part  - Done /  == .99

    # Transaction amount is multiple of 5
def multiple_of_5(data):
    '''
    Creates an indicator of whether or not the transaction amount is multiple of 5
    '''    
    data['TransactionAmt_multiple_5'] = np.remainder(data['TransactionAmt'], 5) == 0
    return data

df.X = multiple_of_5(df.X)
print('Identifying multiples of 5')
print("the shape of X: ", df.X.shape)
print("the shape of y: ", df.y.shape,"\n")

    # Extract decimal part of transaction amount and inidicator variable dec == .99
def decimal_extraction(data):
    '''
    Extracts decimal and integer parts of TransactionAmt
    '''
    data['Decimal_TransactionAmt'] = data['TransactionAmt'] - data['TransactionAmt'].astype(int)
    data['Integer_TransactionAmt'] = data['TransactionAmt'].astype(int)
    return data
        
df.X = decimal_extraction(df.X)
print('Separating decimal and integer parts')
print("the shape of X: ", df.X.shape)
print("the shape of y: ", df.y.shape,"\n")

del df_tot, categorical_cols, numerical_cols, col

#%%
    # Before continuing, we have to split the dataset into test and split. The
    # transformations that we will apply now would lead to leakage if we didn't 
    # treat test and train sets separately from now on
    # We also separate train and dev sets


X = df.X[df.X['Test'] == 0]
X_test = df.X[df.X['Test'] == 1]

y = df.y[df.data['Test'] == 0]

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.1)

    # Standardise input - Standardisation based on training set values
numeric_columns = [] # Select only numeric features first
for col in X_train.columns:
    if X_train[col].dtype!='object' and X_test[col].dtype!='object' and X_dev[col].dtype!='object':
        numeric_columns.append(col)
numeric_columns.remove('TransactionDT') #It doesn't make sense to standardise date
        
scaler = preprocessing.StandardScaler().fit(X_train[numeric_columns]) # Now we can standardise
X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])     
X_dev[numeric_columns] = scaler.transform(X_dev[numeric_columns])

    # Oversample
ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(X_train, y_train)


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

#%%
    # Save model to pickle file
pickle.dump(clf, open('/Users/alvaro.corrales.canoibm.com/Desktop/Scripts/AutoAI internal project' + 
                      "/xgb_model_0202.pickle.dat", "wb"))
#clf = pickle.load(open('/Users/alvaro.corrales.canoibm.com/Desktop/Scripts/AutoAI internal project' + 
#                      "/xgb_model.pickle.dat", "rb"))


#%%
    # Model metrics
y_pred = clf.predict(X_dev)

print('The ROC-AUC score is:', roc_auc_score(y_dev, y_pred))
print('The accuracy score is:', accuracy_score(y_dev, y_pred))
print('The precision score is:', precision_score(y_dev, y_pred))
print('The recall score is:', recall_score(y_dev, y_pred))
print('The F1 score is:', f1_score(y_dev, y_pred))


metrics = {'XGBoost': {'ROC-AUC': roc_auc_score(y_dev, y_pred), 
                       'Accuracy': accuracy_score(y_dev, y_pred),
                       'Precision': precision_score(y_dev, y_pred),
                       'Recall': recall_score(y_dev, y_pred),
                       'F1 score': f1_score(y_dev, y_pred)}}


    ########################
    # FEATURE IMPORTANCE !!!    
    ########################

#%% Prepare submission
sample_submission = pd.read_csv(path + '/sample_submission.csv', 
                                index_col='TransactionID')
sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv(path + '/20200201 - xgb.csv')
