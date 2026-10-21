# -*- coding: utf-8 -*-
"""
Date: Oct 29 20:26:53 2021
@description: This module performs Credit Scoring and Risk Analysis for approving loan to a customer
              The dataset is taken from Kaggle and contains various features 
              regarding the financial conditions of the customer.
              https://www.kaggle.com/karanagarwal/credit-risk-analysis
@author :     Debanjan Saha
@licence:     MIT License
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
import xgboost
from datetime import datetime

from xgboost.core import Booster

data = pd.read_csv('data/Credit_default_dataset.csv')
print(data.head)

data.drop('ID', axis=1, inplace=True)
data.rename(columns={'PAY_0':'PAY_1'}, inplace=True)

print(data.EDUCATION.value_counts())
print(data.MARRIAGE.value_counts())

# In order to reduce the number of unique values, we will incorrectly map some of the values to lower values
data['EDUCATION'] = data['EDUCATION'].map({0:4, 1:1, 2:2, 3:3, 4:4, 5:4, 6:4})
data['MARRIAGE'] = data['MARRIAGE'].map({0:3, 1:1, 2:2, 3:3})
print(data.head())

scaler = StandardScaler()
# Separate independent and dependent variables
x = data.drop('default.payment.next.month', axis=1)
x = scaler.fit_transform(x)
y = data['default.payment.next.month']

# Select Hyperparameters
params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

def timer(start_time = None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

xg_cls = xgboost.XGBClassifier()
rand_s = RandomizedSearchCV(xg_cls, param_distributions=params, n_iter=1, 
                            scoring='roc_auc', n_jobs=1, cv=5, verbose=3)

start_time = timer(None)
rand_s.fit(x, y)
timer(start_time)

print(rand_s.best_estimator_)
        
print(rand_s.best_params_)

# select classifier as suggested by random search CV
classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7,
              enable_categorical=False, gamma=0.2, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=6,
              min_child_weight=5, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1, predictor='auto',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)

score = cross_val_score(classifier, x, y, cv=10)
print(score)
print("Mean Accuracy of Model: ", round(score.mean()*100, 2)) ## 81.61