# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:27:20 2023

@author: Breandán Breathnach
"""

# Loading our libraries
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
import pandas as pd
import statistics
import os
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression
import datetime
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor



# Setting the working directory, and loading our data
os.chdir('C:\\Users\\Breandán Breathnach\\Desktop\\ASDS Git\\ML-group-project')
bicycles = pd.read_csv('BDP_Bikes_Final.csv')

# Viewing our data
bicycles.info()
bicycles.head()
bicycles.dtypes


# Summarising our data
bicycles['bikes'].mean()
bicycles['bikes'].median()
bicycles['bikes'].mode()
bicycles['bikes'].max()
bicycles['bikes'].min()

bicycles['bikes'].value_counts().sort_values(ascending=False).head(20)
bicycles[bicycles['bikes'].isin([40])]



# Making our training and test sets with the COVID-19 dummy
xbicycle_dummy_yes = bicycles[['temp', 'rain', 'workday', 'covid']]
ybicycle_dummy_yes = bicycles['bikes']
xbicycle_train_dummy_yes, xbicycle_test_dummy_yes, ybicycle_train_dummy_yes, ybicycle_test_dummy_yes = train_test_split(xbicycle_dummy_yes, ybicycle_dummy_yes, test_size=0.2, random_state=42)


# LINEAR MODEL #

# Running our Linear Regression model
Linear_model = LinearRegression().fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Ascertaining our linear model's coefficients
print('bikes =', Linear_model.intercept_, '+', Linear_model.coef_[0],'(temp) +', Linear_model.coef_[1], '(rain) +', Linear_model.coef_[2], '(workday) +', Linear_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) of our linear model
lm_pred_train1 = Linear_model.predict(xbicycle_train_dummy_yes)
lm_pred_test1 = Linear_model.predict(xbicycle_test_dummy_yes)

lm_MSE = mean_squared_error(ybicycle_test_dummy_yes, lm_pred_test1)
lm_R2 = r2_score(ybicycle_test_dummy_yes, lm_pred_test1)
print('Linear Regression Mean Squared Error, testing set:', lm_MSE) # = 4538.89
print('Linear Regression R-squared:', lm_R2) # = 0.18

# Predicting the mean value
lm_mean_pred = np.mean(lm_pred_test1)
print("Linear Regression mean predicted value:", lm_mean_pred) # = 69.13


# OLS #

# Running our OLS model

# Adding a constant (requisite for Python OLS)
xbicycle_train_dummy_yes_constant = sm.add_constant(xbicycle_train_dummy_yes)
OLS_model = sm.OLS(ybicycle_train_dummy_yes, xbicycle_train_dummy_yes_constant).fit()

# Ascertaining our OLS model's coefficients
print('bikes =', OLS_model.params[0],'+', OLS_model.params[1], '(temp) +', OLS_model.params[2], '(rain) +', OLS_model.params[3], '(workday) +', OLS_model.params[4], '(covid)')

# Summarising the OLS model
print(OLS_model.summary())

# Calculating the Mean Squared Error (MSE) of our OLS model
ols_pred_train1 = OLS_model.predict(xbicycle_train_dummy_yes_constant)
ols_pred_test1 = OLS_model.predict(sm.add_constant(xbicycle_test_dummy_yes))

ols_MSE = mean_squared_error(ybicycle_test_dummy_yes, ols_pred_test1)
ols_R2 = r2_score(ybicycle_test_dummy_yes, ols_pred_test1)
print('OLS Mean Squared Error, testing set:', ols_MSE) # = 4538.9
print('OLS R-squared:', ols_R2) # = 0.18

# Predicting the mean value
ols_mean_pred = np.mean(ols_pred_test1)
print("OLS mean predicted value:", ols_mean_pred) # = 69.13


# LASSO #

# Running our Lasso model
Lasso_model = LassoCV(alphas=arange(0, 1, 0.01), cv=5, n_jobs=-1)
Lasso_model = Lasso_model.fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes) 

# Ascertaining our linear model's coefficients
print('bikes =', Lasso_model.intercept_ + Lasso_model.coef_[0],'(temp) +', Lasso_model.coef_[1], '(rain) +', Lasso_model.coef_[2], '(workday) +', Lasso_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) of our Lasso model
lasso_pred_train1 = Lasso_model.predict(xbicycle_train_dummy_yes)
lasso_pred_test1 = Lasso_model.predict(xbicycle_test_dummy_yes)

lasso_MSE = mean_squared_error(ybicycle_test_dummy_yes, lasso_pred_test1)
lasso_R2 = r2_score(ybicycle_test_dummy_yes, lasso_pred_test1)
print('Lasso Mean Squared Error, testing set:', lasso_MSE) # = 4538.9
print('Lasso R-squared:', lasso_R2) # = 0.18

# Predicting the mean value
lasso_mean_pred = np.mean(lasso_pred_test1)
print("Exponentiated Lasso mean predicted value:", lasso_mean_pred) # = 69.13


# RIDGE #

# Running our Ridge Regression model
Ridge_model = RidgeCV(alphas=arange(0, 1, 0.01), cv=5, scoring='neg_mean_absolute_error').fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Ascertaining our ridge model's coefficients
print('bikes =', Ridge_model.intercept_ + Ridge_model.coef_[0],'(temp) +', Ridge_model.coef_[1], '(rain) +', Ridge_model.coef_[2], '(workday) +', Ridge_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) of our Ridge model
ridge_pred_train1 = Ridge_model.predict(xbicycle_train_dummy_yes)
ridge_pred_test1 = Ridge_model.predict(xbicycle_test_dummy_yes)
    
ridge_MSE = mean_squared_error(ybicycle_test_dummy_yes, ridge_pred_test1)
ridge_R2 = r2_score(ybicycle_test_dummy_yes, ridge_pred_test1)
print('Ridge Mean Squared Error, testing set:', ridge_MSE) # = 4538.9  
print('Ridge R-squared:', ridge_R2) # = 0.18

# Predicting the mean value
ridge_mean_pred = np.mean(ridge_pred_test1)
print("Ridge mean predicted value:", ridge_mean_pred) # = 69.13


# RANDOM FORESTS #

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) of our Random Forests model
rf_pred_train1 = rf.predict(xbicycle_train_dummy_yes)
rf_pred_test1 = rf.predict(xbicycle_test_dummy_yes)

rf_MSE = mean_squared_error(ybicycle_test_dummy_yes, rf_pred_test1)
rf_R2 = r2_score(ybicycle_test_dummy_yes, rf_pred_test1)
print('Random Forests Mean squared error, testing set:', rf_MSE) # = 4118.25
print('Random Forest R-squared:', rf_R2) # = 0.26

# Predicting the mean value
rf_mean_pred = np.mean(rf_pred_test1)
print("Random Forests mean predicted value:", rf_mean_pred) # = 69.09


# SUPPORT VECTOR #

# Creating our Support Vector Regression model
svr = SVR(kernel='linear', C=1, epsilon=0.1).fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Predict the target variable for the test data
svr_pred_test1 = svr.predict(xbicycle_test_dummy_yes)

# Calculating the Mean Squared Error (MSE) of our Support Vector Regression model
svr_MSE = mean_squared_error(ybicycle_test_dummy_yes, svr_pred_test1)
svr_R2 = r2_score(ybicycle_test_dummy_yes, svr_pred_test1)

print('Support Vector Regression Mean squared error, testing set:', svr_MSE)
print('Support Vector Regression R-squared:', svr_R2)

# Predicting the mean value
svr_mean_pred = np.mean(svr_pred_test1)
print("Support Vector Regression mean predicted value:", svr_mean_pred)


# K NEAREST NEIGHBOURS #

# Creating our KNN model
knn = KNeighborsRegressor(n_neighbors=5).fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) and R-Squared of our KNN model
knn_pred_train1 = knn.predict(xbicycle_train_dummy_yes)
knn_pred_test1 = knn.predict(xbicycle_test_dummy_yes)

knn_MSE = mean_squared_error(ybicycle_test_dummy_yes, knn_pred_test1)
knn_R2 = r2_score(ybicycle_test_dummy_yes, knn_pred_test1)

print('KNN Mean squared error, testing set:', knn_MSE)
print('KNN R-squared:', knn_R2)

# Predicting the mean value
knn_mean_pred = np.mean(knn_pred_test1)
print("KNN mean predicted value:", knn_mean_pred)


# Doing exactly as above, except with polynomial and exponential terms to our equations
bicycles['tempsq'] = bicycles['temp']**2
bicycles['tempcub'] = bicycles['temp']**3
bicycles['tempexp'] = bicycles['temp'].apply(math.exp)

bicycles['rainsq'] = bicycles['rain']**2
bicycles['raincub'] = bicycles['rain']**3
bicycles['rainexp'] = bicycles['rain'].apply(math.exp)


# Making our squared training and testing sets
xbicycle_sq = bicycles[['temp', 'rain', 'workday', 'covid', 'tempsq', 'rainsq']]
xbicycle_train_sq, xbicycle_test_sq, ybicycle_train_dummy_yes, ybicycle_test_dummy_yes = train_test_split(xbicycle_sq, ybicycle_dummy_yes, test_size=0.2, random_state=42)

# Making our squared and cubed training and testing sets
xbicycle_cub = bicycles[['temp', 'rain', 'workday', 'covid', 'tempsq', 'rainsq', 'tempcub', 'raincub']]
xbicycle_train_cub, xbicycle_test_cub, ybicycle_train_dummy_yes, ybicycle_test_dummy_yes = train_test_split(xbicycle_cub, ybicycle_dummy_yes, test_size=0.2, random_state=42)

# Making our squared, cubed, and exponential training and testing sets
xbicycle_exp = bicycles[['temp', 'rain', 'workday', 'covid', 'tempsq', 'rainsq', 'tempcub', 'raincub', 'tempexp', 'rainexp']]
xbicycle_train_exp, xbicycle_test_exp, ybicycle_train_dummy_yes, ybicycle_test_dummy_yes = train_test_split(xbicycle_exp, ybicycle_dummy_yes, test_size=0.2, random_state=42)


# SQUARED # 

# Running our linear model for the squared variables
sqLinear_model = LinearRegression().fit(xbicycle_train_sq, ybicycle_train_dummy_yes)

# Ascertaining our linear model's coefficients
print('bikes =', sqLinear_model.intercept_ + sqLinear_model.coef_[0],'(temp) +', sqLinear_model.coef_[1], '(rain) +', sqLinear_model.coef_[2], '(workday) +', sqLinear_model.coef_[3], '(covid)')

# Calculating the Mean Squared Error (MSE) of our squared linear model
sq_lm_pred_train1 = sqLinear_model.predict(xbicycle_train_sq)
sq_lm_pred_test1 = sqLinear_model.predict(xbicycle_test_sq)

sq_lm_MSE = mean_squared_error(ybicycle_test_dummy_yes, sq_lm_pred_test1)
sq_lm_R2 = r2_score(ybicycle_test_dummy_yes, sq_lm_pred_test1)
print('Squared Linear Regression Mean Squared Error, testing set:', sq_lm_MSE) # = 4494.78
print('Squared Linear Regression R-squared:', sq_lm_R2) # = 0.19

# Predicting the mean value
sq_lm_mean_pred = np.mean(sq_lm_pred_test1)
print("Squared Linear Regression mean predicted value:", sq_lm_mean_pred) # = 69.13


# Running our OLS model for the squared variables

xbicycle_train_sq_constant = sm.add_constant(xbicycle_train_sq)
sqOLS_model = sm.OLS(ybicycle_train_dummy_yes, xbicycle_train_sq_constant).fit()

# Ascertaining our OLS model's coefficients
print('bikes =', sqOLS_model.params[0],'+', sqOLS_model.params[1], '(temp) +', sqOLS_model.params[2], '(rain) +', sqOLS_model.params[3], '(workday) +', sqOLS_model.params[4], '(covid)')

# Summarising the OLS model
print(sqOLS_model.summary())

# Calculating the Mean Squared Error (MSE) of our squared OLS model
sq_ols_pred_train1 = sqOLS_model.predict(xbicycle_train_sq_constant)
sq_ols_pred_test1 = sqOLS_model.predict(sm.add_constant(xbicycle_test_sq))

sq_ols_MSE = mean_squared_error(ybicycle_test_dummy_yes, sq_ols_pred_test1)
sq_ols_R2 = r2_score(ybicycle_test_dummy_yes, sq_ols_pred_test1)
print('Squared OLS Mean Squared Error, testing set:', sq_ols_MSE) # = 4494.78
print('Squared OLS R-squared:', sq_ols_R2) # = 0.19

# Predicting the mean value
sq_ols_mean_pred = np.mean(sq_ols_pred_test1)
print("Squared OLS mean predicted value:", sq_ols_mean_pred) # = 69.13


# Running our Lasso model for the squared variables
sqLasso_model = LassoCV(alphas=arange(0, 1, 0.01), cv=5, n_jobs=-1)
sqLasso_model = sqLasso_model.fit(xbicycle_train_sq, ybicycle_train_dummy_yes)

# Ascertaining our Lasso model's coefficients
print('bikes =', sqLasso_model.intercept_ + sqLasso_model.coef_[0],'(temp) +', sqLasso_model.coef_[1], '(rain) +', sqLasso_model.coef_[2], '(workday) +', sqLasso_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) of our squared Lasso model
sq_lasso_pred_train1 = sqLasso_model.predict(xbicycle_train_sq)
sq_lasso_pred_test1 = sqLasso_model.predict(xbicycle_test_sq)

sq_lasso_MSE = mean_squared_error(ybicycle_test_dummy_yes, sq_lasso_pred_test1)
sq_lasso_R2 = r2_score(ybicycle_test_dummy_yes, sq_lasso_pred_test1)
print('Squared Lasso Mean Squared Error, testing set:', sq_lasso_MSE) # = 4494.75
print('Squared Lasso R-squared:', sq_lasso_R2) # = 0.19


# Predicting the mean value
sq_lasso_mean_pred = np.mean(sq_lasso_pred_test1)
print("Squared Lasso mean predicted value:", sq_lasso_mean_pred) # = 69.13


# Running our Ridge model for the squared variables
sqRidge_model = RidgeCV(alphas=arange(0, 1, 0.01), cv=5, scoring='neg_mean_absolute_error').fit(xbicycle_train_sq, ybicycle_train_dummy_yes)

# Ascertaining our squared ridge model's coefficients
print('bikes =', sqRidge_model.intercept_ + sqRidge_model.coef_[0],'(temp) +', sqRidge_model.coef_[1], '(rain) +', sqRidge_model.coef_[2], '(workday) +', sqRidge_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) of our squared Ridge model
sq_ridge_pred_train1 = sqRidge_model.predict(xbicycle_train_sq)
sq_ridge_pred_test1 = sqRidge_model.predict(xbicycle_test_sq)

sq_ridge_MSE = mean_squared_error(ybicycle_test_dummy_yes, sq_ridge_pred_test1)
sq_ridge_R2 = r2_score(ybicycle_test_dummy_yes, sq_ridge_pred_test1)
print('Squared Ridge Mean Squared Error, testing set:', sq_ridge_MSE) # = 4494.78
print('Squared Ridge R-squared:', sq_ridge_R2) # = 0.19

# Predicting the mean value
sq_ridge_mean_pred = np.mean(sq_ridge_pred_test1)
print("Squared Ridge mean predicted value:", sq_ridge_mean_pred) # = 69.13


# Running our Random Forests model for the squared variables
sq_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(xbicycle_train_sq, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) of our Random Forests model
sq_rf_pred_train1 = sq_rf.predict(xbicycle_train_sq)
sq_rf_pred_test1 = sq_rf.predict(xbicycle_test_sq)

sq_rf_MSE = mean_squared_error(ybicycle_test_dummy_yes, sq_rf_pred_test1)
sq_rf_R2 = r2_score(ybicycle_test_dummy_yes, sq_rf_pred_test1)
print('Squared Random Forests Mean squared error, testing set:', sq_rf_MSE) # = 4118.25
print('Squared Random Forest R-squared:', sq_rf_R2) # = 0.26

# Predicting the mean value
sq_rf_mean_pred = np.mean(sq_rf_pred_test1)
print("Squared Random Forests mean predicted value:", sq_rf_mean_pred) # = 69.09

print(rf.feature_importances_)
R
# Creating our squared Support Vector Regression model
sq_svr = SVR(kernel='linear', C=1, epsilon=0.1).fit(xbicycle_train_sq, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) and R-Squared of our KNN Regression model
sq_svr_pred_train1 = sq_svr.predict(xbicycle_train_sq)
sq_svr_pred_test1 = sq_svr.predict(xbicycle_test_sq)

sq_svr_MSE = mean_squared_error(ybicycle_test_dummy_yes, sq_svr_pred_test1)
sq_svr_R2 = r2_score(ybicycle_test_dummy_yes, sq_svr_pred_test1)

print('Squared Support Vector Regression Mean squared error, testing set:', sq_svr_MSE)
print('Squared Support Vector Regression R-squared:', sq_svr_R2)

# Predicting the mean value
sq_svr_mean_pred = np.mean(sq_svr_pred_test1)
print("Squared Support Vector Regression mean predicted value:", sq_svr_mean_pred)


# Creating our squared KNN model
sq_knn = KNeighborsRegressor(n_neighbors=5).fit(xbicycle_train_sq, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) and R-Squared of our squared KNN model
sq_knn_pred_train1 = sq_knn.predict(xbicycle_train_sq)
sq_knn_pred_test1 = sq_knn.predict(xbicycle_test_sq)

sq_knn_MSE = mean_squared_error(ybicycle_test_dummy_yes, sq_knn_pred_test1)
sq_knn_R2 = r2_score(ybicycle_test_dummy_yes, sq_knn_pred_test1)

print('Squared KNN Mean squared error, testing set:', sq_knn_MSE)
print('Squared KNN R-squared:', sq_knn_R2)

# Predicting the mean value
sq_knn_mean_pred = np.mean(sq_knn_pred_test1)
print("Squared KNN mean predicted value:", sq_knn_mean_pred)



# CUBED #

# Running our linear model for the squared and cubed variables
cubLinear_model = LinearRegression().fit(xbicycle_train_cub, ybicycle_train_dummy_yes)

# Ascertaining our cubed linear model's coefficients
print('bikes =', cubLinear_model.intercept_ + cubLinear_model.coef_[0],'(temp) +', cubLinear_model.coef_[1], '(rain) +', cubLinear_model.coef_[2], '(workday) +', cubLinear_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) of our cubed linear model
cub_lm_pred_train1 = cubLinear_model.predict(xbicycle_train_cub)
cub_lm_pred_test1 = cubLinear_model.predict(xbicycle_test_cub)

cub_lm_MSE = mean_squared_error(ybicycle_test_dummy_yes, cub_lm_pred_test1)
cub_lm_R2 = r2_score(ybicycle_test_dummy_yes, cub_lm_pred_test1)
print('Cubed Linear Regression Mean Squared Error, testing set:', cub_lm_MSE) # = 4488.55
print('Cubed Linear Regression R-squared:', cub_lm_R2) # = 0.19


# Predicting the mean value
cub_lm_mean_pred = np.mean(cub_lm_pred_test1)
print("Cubed Linear Regression mean predicted value:", cub_lm_mean_pred) # = 69.12


# Running our OLS model for the squared and cubed variables
xbicycle_train_cub_constant = sm.add_constant(xbicycle_train_cub)
cubOLS_model = sm.OLS(ybicycle_train_dummy_yes, xbicycle_train_cub_constant).fit()

# Ascertaining our cubed OLS model's coefficients
print('bikes =', cubOLS_model.params[0],'+', cubOLS_model.params[1], '(temp) +', cubOLS_model.params[2], '(rain) +', cubOLS_model.params[3], '(workday) +', cubOLS_model.params[4], '(covid)')

# Summarising the cubed OLS model
print(cubOLS_model.summary())

# Calculating the Mean Squared Error (MSE) of our cubed OLS model
cub_ols_pred_train1 = cubOLS_model.predict(xbicycle_train_cub_constant)
cub_ols_pred_test1 = cubOLS_model.predict(sm.add_constant(xbicycle_test_cub))

cub_ols_MSE = mean_squared_error(ybicycle_test_dummy_yes, cub_ols_pred_test1)
cub_ols_R2 = r2_score(ybicycle_test_dummy_yes, cub_ols_pred_test1)
print('Cubed OLS Mean Squared Error, testing set:', cub_ols_MSE) # = 4488.55
print('Cubed OLS Regression R-squared:', cub_ols_R2) # = 0.19

# Predicting the mean value
cub_ols_mean_pred = np.mean(cub_ols_pred_test1)
print("Cubed OLS mean predicted value:", cub_ols_mean_pred) # = 69.12


# Running our Lasso model for the squared and cubed variables
cubLasso_model = LassoCV(alphas=arange(0, 1, 0.01), cv=5, n_jobs=-1)
cubLasso_model = cubLasso_model.fit(xbicycle_train_cub, ybicycle_train_dummy_yes)

# Ascertaining our Lasso model's coefficients
print('bikes =', cubLasso_model.intercept_ + cubLasso_model.coef_[0],'(temp) +', cubLasso_model.coef_[1], '(rain) +', cubLasso_model.coef_[2], '(workday) +', cubLasso_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) of our cubed Lasso model
cub_lasso_pred_train1 = cubLasso_model.predict(xbicycle_train_cub)
cub_lasso_pred_test1 = cubLasso_model.predict(xbicycle_test_cub)

cub_lasso_MSE = mean_squared_error(ybicycle_test_dummy_yes, cub_lasso_pred_test1)
cub_lasso_R2 = r2_score(ybicycle_test_dummy_yes, cub_lasso_pred_test1)
print('Cubed Lasso Mean Squared Error, testing set:', cub_lasso_MSE) # = 4488.55
print('Cubed Lasso Regression R-squared:', cub_lasso_R2) # = 0.19


# Predicting the mean value
cub_lasso_mean_pred = np.mean(cub_lasso_pred_test1)
print("Cubed Lasso mean predicted value:", cub_lasso_mean_pred) # = 69.12


# Running our Ridge model for the squared and cubed variables
cubRidge_model = RidgeCV(alphas=arange(0, 1, 0.01), cv=5, scoring='neg_mean_absolute_error').fit(xbicycle_train_cub, ybicycle_train_dummy_yes)

# Ascertaining our ridge model's coefficients
print('bikes =', cubRidge_model.intercept_ + cubRidge_model.coef_[0],'(temp) +', cubRidge_model.coef_[1], '(rain) +', cubRidge_model.coef_[2], '(workday) +', cubRidge_model.coef_[3], '(covid)')

# Summarising the cubed Ridge model
cub_ridge_pred_train1 = cubRidge_model.predict(xbicycle_train_cub)
cub_ridge_pred_test1 = cubRidge_model.predict(xbicycle_test_cub)

cub_ridge_MSE = mean_squared_error(ybicycle_test_dummy_yes, cub_ridge_pred_test1)
cub_ridge_R2 = r2_score(ybicycle_test_dummy_yes, cub_ridge_pred_test1)
print('Cubed Ridge Mean Squared Error, testing set:', cub_ridge_MSE) # = 4488.55
print('Cubed Ridge Regression R-squared:', cub_ridge_R2) # = 0.19

# Predicting the mean value
cub_ridge_mean_pred = np.mean(cub_ridge_pred_test1)
print("Cubed Ridge mean predicted value:", cub_ridge_mean_pred) # = 69.12


# Running our Random Forests model for the cubed variables
cub_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(xbicycle_train_cub, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) and R-squared of our Random Forests model
cub_rf_pred_train1 = cub_rf.predict(xbicycle_train_cub)
cub_rf_pred_test1 = cub_rf.predict(xbicycle_test_cub)

cub_rf_MSE = mean_squared_error(ybicycle_test_dummy_yes, cub_rf_pred_test1)
cub_rf_R2 = r2_score(ybicycle_test_dummy_yes, cub_rf_pred_test1)
print('Cubed Random Forests Mean squared error, testing set:', cub_rf_MSE) # = 4122.35
print('Cubed Random Forest R-squared:', cub_rf_R2) # = 0.26

# Predicting the mean value
cub_rf_mean_pred = np.mean(cub_rf_pred_test1)
print("Cubed Random Forests mean predicted value:", cub_rf_mean_pred) # = 69.09


# Creating our squared Support Vector Regression model
cub_svr = SVR(kernel='linear', C=1, epsilon=0.1).fit(xbicycle_train_cub, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) of our Support Vector Regression model
cub_svr_pred_train1 = cub_svr.predict(xbicycle_train_cub)
cub_svr_pred_test1 = cub_svr.predict(xbicycle_test_cub)

cub_svr_MSE = mean_squared_error(ybicycle_test_dummy_yes, cub_svr_pred_test1)
cub_svr_R2 = r2_score(ybicycle_test_dummy_yes, cub_svr_pred_test1)

print('Cubed Support Vector Regression Mean squared error, testing set:', cub_svr_MSE)
print('Cubed Support Vector Regression R-squared:', cub_svr_R2)

# Predicting the mean value
cub_svr_mean_pred = np.mean(cub_svr_pred_test1)
print("Cubed Support Vector Regression mean predicted value:", cub_svr_mean_pred)


# Creating our cubed KNN model
cub_knn = KNeighborsRegressor(n_neighbors=5).fit(xbicycle_train_cub, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) and R-Squared of our cubed KNN model
cub_knn_pred_train1 = cub_knn.predict(xbicycle_train_cub)
cub_knn_pred_test1 = cub_knn.predict(xbicycle_test_cub)

cub_knn_MSE = mean_squared_error(ybicycle_test_dummy_yes, cub_knn_pred_test1)
cub_knn_R2 = r2_score(ybicycle_test_dummy_yes, cub_knn_pred_test1)

print('Cubed KNN Mean squared error, testing set:', cub_knn_MSE)
print('Cubed KNN R-squared:', cub_knn_R2)

# Predicting the mean value
cub_knn_mean_pred = np.mean(cub_knn_pred_test1)
print("Cubed KNN mean predicted value:", cub_knn_mean_pred)


# EXPONENTIAL #

# Running our linear model for the squared, cubed, and exponential variables
expLinear_model = LinearRegression().fit(xbicycle_train_exp, ybicycle_train_dummy_yes)

# Ascertaining our linear model's coefficients
print('bikes =', expLinear_model.intercept_ + expLinear_model.coef_[0],'(temp) +', expLinear_model.coef_[1], '(rain) +', expLinear_model.coef_[2], '(workday) +', expLinear_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) and R-Squared of our exponentiated linear model
exp_lm_pred_train1 = expLinear_model.predict(xbicycle_train_exp)
exp_lm_pred_test1 = expLinear_model.predict(xbicycle_test_exp)

exp_lm_MSE = mean_squared_error(ybicycle_test_dummy_yes, exp_lm_pred_test1)
exp_lm_R2 = r2_score(ybicycle_test_dummy_yes, exp_lm_pred_test1)
print('Exponentiated Linear Regression Mean Squared Error, testing set:', exp_lm_MSE) # = 4485.13
print('Exponentiated Linear Regression R-squared:', exp_lm_R2) # = 0.19

# Predicting the mean value
exp_lm_mean_pred = np.mean(exp_lm_pred_test1)
print("Exponentiated Linear Regression mean predicted value:", exp_lm_mean_pred) # = 69.13


# Running our OLS model for the squared, cubed, and exponential variables
xbicycle_train_exp_constant = sm.add_constant(xbicycle_train_exp)
expOLS_model = sm.OLS(ybicycle_train_dummy_yes, xbicycle_train_exp_constant).fit()

# Ascertaining our OLS model's coefficients
print('bikes =', expOLS_model.params[0],'+', expOLS_model.params[1], '(temp) +', expOLS_model.params[2], '(rain) +', expOLS_model.params[3], '(workday) +', expOLS_model.params[4], '(covid)')

# Summarising the exponentiated OLS model
print(expOLS_model.summary())


# Calculating the Mean Squared Error (MSE) and R-Squared of our exponentiated OLS model
exp_ols_pred_train1 = expOLS_model.predict(xbicycle_train_exp_constant)
exp_ols_pred_test1 = expOLS_model.predict(sm.add_constant(xbicycle_test_exp))

exp_ols_MSE = mean_squared_error(ybicycle_test_dummy_yes, exp_ols_pred_test1)
exp_ols_R2 = r2_score(ybicycle_test_dummy_yes, exp_ols_pred_test1)
print('Exponentiated OLS Mean Squared Error, testing set:', exp_ols_MSE) # = 4485.13
print('Exponentiated OLS R-squared:', exp_ols_R2) # = 0.19

# Predicting the mean value
exp_ols_mean_pred = np.mean(exp_ols_pred_test1)
print("Exponentiated OLS mean predicted value:", exp_ols_mean_pred) # = 69.13


# Running our Lasso model for the squared, cubed, and exponential variables
expLasso_model = LassoCV(alphas=arange(0, 1, 0.01), cv=5, n_jobs=-1)
expLasso_model = expLasso_model.fit(xbicycle_train_exp, ybicycle_train_dummy_yes)

# Ascertaining our Lasso model's coefficients
print('bikes =', expLasso_model.intercept_ + expLasso_model.coef_[0],'(temp) +', expLasso_model.coef_[1], '(rain) +', expLasso_model.coef_[2], '(workday) +', expLasso_model.coef_[3], '(covid)')


# Calculating the Mean Squared Error (MSE) and R-Squared of our exponentiated Lasso model
exp_lasso_pred_train1 = expLasso_model.predict(xbicycle_train_exp)
exp_lasso_pred_test1 = expLasso_model.predict(xbicycle_test_exp)

exp_lasso_MSE = mean_squared_error(ybicycle_test_dummy_yes, exp_lasso_pred_test1)
exp_lasso_R2 = r2_score(ybicycle_test_dummy_yes, exp_lasso_pred_test1)
print('Exponentiated Lasso Mean Squared Error, testing set:', exp_lasso_MSE) # = 4485.12
print('Exponentiated Lasso R-squared:', exp_lasso_R2) # = 0.19

# Predicting the mean value
exp_lasso_mean_pred = np.mean(exp_lasso_pred_test1)
print("Exponentiated Lasso mean predicted value:", exp_lasso_mean_pred) # = 69.13


# Running our Ridge model for the squared, cubed, and exponential variables
expRidge_model = RidgeCV(alphas=arange(0, 1, 0.01), cv=5, scoring='neg_mean_absolute_error').fit(xbicycle_train_exp, ybicycle_train_dummy_yes)

# Ascertaining our ridge model's coefficients
print('bikes =', expRidge_model.intercept_ + expRidge_model.coef_[0],'(temp) +', expRidge_model.coef_[1], '(rain) +', expRidge_model.coef_[2], '(workday) +', expRidge_model.coef_[3], '(covid)')

# Calculating the Mean Squared Error (MSE) and R-Squared of our exponentiated Ridge model
exp_ridge_pred_train1 = expRidge_model.predict(xbicycle_train_exp)
exp_ridge_pred_test1 = expRidge_model.predict(xbicycle_test_exp)

exp_ridge_MSE = mean_squared_error(ybicycle_test_dummy_yes, exp_ridge_pred_test1)
exp_ridge_R2 = r2_score(ybicycle_test_dummy_yes, exp_ridge_pred_test1)
print('Exponentiated Ridge Mean Squared Error, testing set:', exp_ridge_MSE) # = 4485.12
print('Exponentiated Ridge R-squared:', exp_ridge_R2) # = 0.19

# Predicting the mean value
exp_ridge_mean_pred = np.mean(exp_ridge_pred_test1)
print("Exponentiated Ridge mean predicted value:", exp_ridge_mean_pred) # = 69.13


# Running our Random Forests model for the exponentiated variables
exp_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(xbicycle_train_exp, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) and R-Squared of our Random Forests model
exp_rf_pred_train1 = exp_rf.predict(xbicycle_train_exp)
exp_rf_pred_test1 = exp_rf.predict(xbicycle_test_exp)

exp_rf_MSE = mean_squared_error(ybicycle_test_dummy_yes, exp_rf_pred_test1)
exp_rf_R2 = r2_score(ybicycle_test_dummy_yes, exp_rf_pred_test1)
print('Expoentiated Random Forests Mean squared error, testing set:', exp_rf_MSE) # = 4122.35
print('Exponentiated Random Forest R-squared:', exp_rf_R2) # = 0.26

# Predicting the mean value
exp_rf_mean_pred = np.mean(exp_rf_pred_test1)
print("Exponentiated Random Forests mean predicted value:", exp_rf_mean_pred) # = 69.09


# Creating our squared Support Vector Regression model
exp_svr = SVR(kernel='linear', C=1, epsilon=0.1).fit(xbicycle_train_exp, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) of our Support Vector Regression model
exp_svr_pred_train1 = exp_svr.predict(xbicycle_train_exp)
exp_svr_pred_test1 = exp_svr.predict(xbicycle_test_exp)

exp_svr_MSE = mean_squared_error(ybicycle_test_dummy_yes, exp_svr_pred_test1)
exp_svr_R2 = r2_score(ybicycle_test_dummy_yes, exp_svr_pred_test1)

print('Exponentiated Support Vector Regression Mean squared error, testing set:', exp_svr_MSE)
print('Exponentiated Support Vector Regression R-squared:', exp_svr_R2)

# Predicting the mean value
exp_svr_mean_pred = np.mean(exp_svr_pred_test1)
print("Exponentiated Support Vector Regression mean predicted value:", exp_svr_mean_pred)


# Creating our exponentiated KNN model
exp_knn = KNeighborsRegressor(n_neighbors=5).fit(xbicycle_train_exp, ybicycle_train_dummy_yes)

# Calculating the Mean Squared Error (MSE) and R-Squared of our exponentiated KNN model
exp_knn_pred_train1 = exp_knn.predict(xbicycle_train_exp)
exp_knn_pred_test1 = exp_knn.predict(xbicycle_test_exp)

exp_knn_MSE = mean_squared_error(ybicycle_test_dummy_yes, exp_knn_pred_test1)
exp_knn_R2 = r2_score(ybicycle_test_dummy_yes, exp_knn_pred_test1)

print('Exponentiated KNN Mean squared error, testing set:', exp_knn_MSE)
print('Exponentiated KNN R-squared:', exp_knn_R2)

# Predicting the mean value
exp_knn_mean_pred = np.mean(exp_knn_pred_test1)
print("Exponentiated KNN mean predicted value:", exp_knn_mean_pred)