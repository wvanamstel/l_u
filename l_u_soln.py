# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 10:21:22 2015

@author: w
"""

import pandas as pd
import numpy as np
import urllib
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score


def retrieve_files():
    '''Download training and testing set and unzip in current directory
    IN: none
    OUT: files to disk
    '''
    # Download training and testing set
    print 'Downloading training and testing files\n'
    urllib.urlretrieve('https://resources.lendingclub.com/LoanStats3b.csv.zip',
                       'LoanStats3b.csv.zip')
    urllib.urlretrieve('https://resources.lendingclub.com/LoanStats3c.csv.zip',
                       'LoanStats3c.csv.zip')
    
    # Unzip files
    print 'Unzipping files\n'
    zip_train = zipfile.ZipFile('LoanStats3b.csv.zip')
    zip_test = zipfile.ZipFile('LoanStats3c.csv.zip')
    zip_train.extract('LoanStats3b.csv')
    zip_test.extract('LoanStats3c.csv')
    
    # Close zipfiles
    print 'Done'
    zip_train.close()
    zip_test.close()
    
def read_data():
    '''Read the unzipped csv files from disk
    IN: none
    OUT: dataframe; training and test set
    '''
    train = pd.read_csv('LoanStats3b.csv', delimiter=',', header=1)
    test = pd.read_csv('LoanStats3c.csv', delimiter=',', header=1)
    
    return train, test
    
def preprocess_data(df):
    '''Preprocess data, select columns to be used in the model
    IN: dataframe; raw data with all features and response variable
    OUT: selected features and engineered extra features
    '''
    # Remove any rows which have no loan status and encode a new response col
    # where 'Fully Paid' == 1, and the rest == 0
    df = df[pd.notnull(df.loan_status)]
    response = df.loan_status.apply(lambda x: 1 if x=='Fully Paid' else 0)
    df['response'] = response
    
    # Remove current loan data
    df = df[df.loan_status!='Current']
    
    # Clean up data so that it is model readable, encode categorical variables
    df.loc[:, 'term'] = df.loc[:,'term'].apply(lambda x: float(str(x)[1:3]))
    df.loc[:, 'int_rate'] = df.loc[:, 'int_rate'].apply(lambda x: float(str(x).strip('%')))
    df.loc[:, 'is_inc_v'] = df.loc[:, 'is_inc_v'].apply(lambda x: 0 if x=='Not Verified' else 1)
    df.loc[:, 'pymnt_plan'] = df.loc[:, 'pymnt_plan'].apply(lambda x: 0 if x=='n' else 1)
    df.loc[:, 'zip_code'] = df.loc[:, 'zip_code'].apply(lambda x: float(x[:-2]))
    df.loc[:, 'earliest_cr_line'] = df.loc[:, 'earliest_cr_line'].apply(lambda x: float(x[-4:]))
    df.loc[:, 'revol_util'] = df.loc[:, 'revol_util'].apply(lambda x: float(str(x).strip('%'))/100)
    df.loc[:, 'revol_util'] = df.loc[:, 'revol_util'].apply(lambda x: 1 if x>0.9 else 0)
    
    ### Feature engineering
    # last payment less than monthly reqrd installment 
    # --> borower may be falling behind with payments
    feat1 = (df.installment - df.last_pymnt_amnt).apply(lambda x: 1 if x>=0 else 0)
    df['behind'] = feat1
    # borrowers are not behind if they paid less than the installment,
    # but there is no new scheduled payment, meaning the loan in paid
    idx = pd.isnull(df.next_pymnt_d)
    df.behind[idx] = 0
    
    # DTI: according to http://www.bankrate.com/brm/news/mortgages/20070116_debt_income_ratio_a1.asp
    # a debt to income ratio > 36 is getting into risky territory, encode the dti column accordingly:
    # '1' for a risky dti, else '0'
    df.loc[:, 'dti'] = df.loc[:, 'dti'].apply(lambda x: 1 if x > 40. else 0)
    
    # Calculate ratio of loan amount and annual income
    # Encode a as a binary value, depending on arbitrary risk level
    df['loan_to_income'] = df.loan_amnt/df.annual_inc
    df.loc[:, 'loan_to_income'] = df.loc[:, 'loan_to_income'].apply(lambda x: 1 if x>0.25 else 0)
    
    # These are the columns to be used in the model
    cols_small = ['term', 'loan_amnt', 'int_rate', 'zip_code', 
                      'pymnt_plan', 'response', 'annual_inc', 'dti', 'behind',
                      'revol_util', 'is_inc_v', 'delinq_2yrs', 
                      'collections_12_mths_ex_med', 'loan_to_income']
    
    
    # Subset the features used for the model
    df = df[cols_small]
    
    # Encode any categorical variables
    df = pd.get_dummies(df)
    
    # Drop missing values
    df = df.dropna()

    return df
    
def run_model(df):
    '''
    Runs a random forest classifier on the features
    IN: dataframe; features and response
    OUT: classifier
    '''
    #Split the response and the features
    y = df.pop('response')
    X = df.values
    
    # Split into training/validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    # Fit the random forest classifier                               
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    # Print the metrics
    print 'Training set metrics:\n'
    print 'Confusion matrix:\n'
    print confusion_matrix(y_test, preds)
    
    print 'Approval %: ', sum(preds)/float(preds.shape[0])
    print 'Repay Precision: ', precision_score(y_test, preds)
    
    return clf
    
def predict(clf, df):
    ''' Predict response from features given a classifier
    IN: classifier; model to be used for predictions
        dataframe; features and responses
    OUT: predictions and true responses
    '''
    # Split the data into features and response columns
    y_true = df.pop('response')
    X = df.values
    
    # Make predictions
    preds = clf.predict(X)
    
    return preds, y_true

def print_output(preds, y, ind):
    ''' Print results and metrics to stdout, as well as predictions to file
    IN: np.array; predictions from model
        np.array: true response values
        ind: indices of response values wrt test set
    OUT: stdout: prediction results
         file: prediction results
    '''
    # Construct a list with predictions we can iterate over
    list_pr = []
    for result in preds:
        if result==1:
            list_pr.append('Fully Paid')
        else:
            list_pr.append('NOT Fully Paid')    
            
    # Calculate evaluation metrics
    repay = precision_score(y, preds)
    approv = sum(preds)/float(preds.shape[0])
    metric = repay * approv
    
    # Output to stdout
    for i, ind in enumerate(ind):
       print 'Row Index: ', ind, ' Classification: ', list_pr[i]
       
    print '\n\nApproval %: ', approv
    print 'Repay Precision: ', repay
    print 'Metric: ', metric
    
    print '\n\nNumber of loans Fully Paid: ', sum(preds)
    print 'Number of loans NOT Fully Paid: ', preds.shape[0] - sum(preds)
    
    # Write predictions to file
    pr_df = pd.DataFrame(list_pr)
    pr_df.to_csv('predictions.csv', header=None, index=False)
    
if __name__ == '__main__':
    print 'Retrieving files...'
    retrieve_files()    # Retrieve files
    print 'Loading training and testing set...'
    train, test = read_data()  # Load train and test set and 
    print 'Preprocessing training and testing set...'
    df_train = preprocess_data(train)  # Preprocess training set
    df_test = preprocess_data(test)  # Preprocess test set
    print 'Running model...'
    clf = run_model(df_train)  # Fit classifier model
    preds, y_true = predict(clf, df_test)  # Make predictions
    print_output(preds, y_true, list(df_test.index))  # Print results
    