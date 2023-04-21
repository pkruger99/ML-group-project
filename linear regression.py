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
os.chdir('C:\\Users\phili\Downloads\ML-group-project-main\ML-group-project-main')
bicycles = pd.read_csv('comp2.csv')

bicycles['TIME'] = pd.to_datetime(bicycles['TIME'])
print(bicycles['TIME'].dtype)
print(bicycles['bikes'].dtype)
plt.plot(bicycles['TIME'], bicycles['bikes'])
plt.title("Full time period")
plt.show()

bicycles['bikes'].value_counts().sort_values(ascending=False).head(20)
bicycles[bicycles['bikes'].isin([40])]

filtered_df5 = bicycles.loc[50796:51075]
filtered_df6 = bicycles.loc[52803:54788]

filtered_df8 = bicycles.loc[:67338]
filtered_df9 = bicycles.loc[67338:]

print(filtered_df8["bikes"].mean())
print(filtered_df9["bikes"].mean())

plt.plot(filtered_df5['TIME'], filtered_df5['bikes'])
plt.title("Daily Cycle")
plt.show()

plt.plot(filtered_df6['TIME'], filtered_df6['bikes'])
plt.title("Weekly Cycle")
plt.show()
# Making our training and test sets with the COVID-19 dummy
xbicycle_dummy_yes = bicycles[['temp', 'rain', 'workday', 'covid',"wind"]]
ybicycle_dummy_yes = bicycles['bikes']
xbicycle_train_dummy_yes, xbicycle_test_dummy_yes, ybicycle_train_dummy_yes, ybicycle_test_dummy_yes = train_test_split(xbicycle_dummy_yes, ybicycle_dummy_yes, test_size=0.2, random_state=42)


# LINEAR MODEL #

# Running our Linear Regression model
Linear_model = LinearRegression().fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Ascertaining our linear model's coefficients
print('bikes =', Linear_model.intercept_, '+', Linear_model.coef_[0],'(temp) +', Linear_model.coef_[1], '(rain) +', Linear_model.coef_[2], '(workday) +', Linear_model.coef_[3], '(covid) +',Linear_model.coef_[4],'(wind)')

lm_pred_train1 = Linear_model.predict(xbicycle_train_dummy_yes)
lm_pred_test1 = Linear_model.predict(xbicycle_test_dummy_yes)

lm_MSE = mean_squared_error(ybicycle_test_dummy_yes, lm_pred_test1)
lm_R2 = r2_score(ybicycle_test_dummy_yes, lm_pred_test1)
print('Linear Regression Mean Squared Error, testing set:', lm_MSE) # = 4538.89
print('Linear Regression R-squared:', lm_R2) # = 0.18

reg_list = []
for i in range(0,bicycles.shape[0]):
    reg_list.append(-3.3825124263315445 + 3.736556075597166*bicycles.iloc[i,4] - 10.424736737059334*bicycles.iloc[i,5] + 28.163406408803628*bicycles.iloc[i,12] -26.75400635356541*bicycles.iloc[i,11]+ 1.8196342251435915*bicycles.iloc[i,6]+1.6326834776479986*bicycles.iloc[i,1])
    
plt.plot(bicycles['TIME'], bicycles['bikes'])
plt.plot(bicycles['TIME'], reg_list, alpha = 0.7)
plt.title("day and night")
plt.show()

filtered_df = bicycles.loc[(bicycles['HOUR'] > 7) & (bicycles['HOUR'] < 21)]

xbicycle_dummy_yes = filtered_df[['temp', 'rain', 'workday', 'covid',"wind"]]
ybicycle_dummy_yes = filtered_df['bikes']
xbicycle_train_dummy_yes, xbicycle_test_dummy_yes, ybicycle_train_dummy_yes, ybicycle_test_dummy_yes = train_test_split(xbicycle_dummy_yes, ybicycle_dummy_yes, test_size=0.2, random_state=42)


# LINEAR MODEL #

# Running our Linear Regression model
Linear_model = LinearRegression().fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Ascertaining our linear model's coefficients
print('bikes =', Linear_model.intercept_, '+', Linear_model.coef_[0],'(temp) +', Linear_model.coef_[1], '(rain) +', Linear_model.coef_[2], '(workday) +', Linear_model.coef_[3], '(covid) +',Linear_model.coef_[4],'(wind)')

lm_pred_train1 = Linear_model.predict(xbicycle_train_dummy_yes)
lm_pred_test1 = Linear_model.predict(xbicycle_test_dummy_yes)

lm_MSE = mean_squared_error(ybicycle_test_dummy_yes, lm_pred_test1)
lm_R2 = r2_score(ybicycle_test_dummy_yes, lm_pred_test1)
print('Linear Regression Mean Squared Error, testing set:', lm_MSE) # = 4538.89
print('Linear Regression R-squared:', lm_R2) # = 0.18

reg_list = []
for i in range(0,filtered_df.shape[0]):
    reg_list.append(53.62117296751988 + 3.5851503391942616*filtered_df.iloc[i,4] - 12.600695012797239*filtered_df.iloc[i,5] + 37.058611808909255*filtered_df.iloc[i,12] -43.84066475817235*filtered_df.iloc[i,11]+ 1.6391592130255948*filtered_df.iloc[i,6])
    
plt.plot(filtered_df['TIME'], filtered_df['bikes'])
plt.plot(filtered_df['TIME'], reg_list, alpha = 0.7)
plt.title("day")
plt.show()




filtered_df2 = bicycles.loc[103638:]
filtered_df2.fillna(0, inplace=True)
print("\n")

xbicycle_dummy_yes = filtered_df2[['temp', 'rain', 'workday', 'covid',"wind","bikes +1 week","bikes +1 year","bikes +1 month"]]
ybicycle_dummy_yes = filtered_df2['bikes']
print(xbicycle_dummy_yes.shape[0])
print(ybicycle_dummy_yes.shape[0])
xbicycle_train_dummy_yes, xbicycle_test_dummy_yes, ybicycle_train_dummy_yes, ybicycle_test_dummy_yes = train_test_split(xbicycle_dummy_yes, ybicycle_dummy_yes, test_size=0.2, random_state=42)


# LINEAR MODEL #

# Running our Linear Regression model
Linear_model = LinearRegression().fit(xbicycle_train_dummy_yes, ybicycle_train_dummy_yes)

# Ascertaining our linear model's coefficients
print('bikes =', Linear_model.intercept_, '+', Linear_model.coef_[0],'(temp) +', Linear_model.coef_[1], '(rain) +', Linear_model.coef_[2], '(workday) +', Linear_model.coef_[3], '(covid) +',Linear_model.coef_[4],'(wind) +',Linear_model.coef_[5],'(week) +',Linear_model.coef_[6],'(year) +',Linear_model.coef_[7],'(month)')


lm_pred_train1 = Linear_model.predict(xbicycle_train_dummy_yes)
lm_pred_test1 = Linear_model.predict(xbicycle_test_dummy_yes)

lm_MSE = mean_squared_error(ybicycle_test_dummy_yes, lm_pred_test1)
lm_R2 = r2_score(ybicycle_test_dummy_yes, lm_pred_test1)
print('Linear Regression Mean Squared Error, testing set:', lm_MSE) # = 4538.89
print('Linear Regression R-squared:', lm_R2) # = 0.18


reg_list = []
for i in range(0,filtered_df2.shape[0]):
    reg_list.append(-16.236957954611455 + 2.128572641174578*filtered_df2.iloc[i,4] -9.501432754760447*filtered_df2.iloc[i,5] + 16.299968239470523*filtered_df2.iloc[i,12] -6.217248937900877e-14*filtered_df2.iloc[i,11]+ + 0.9627233018504205*filtered_df2.iloc[i,6] + 0.26642792530659776*filtered_df2.iloc[i,8]+ 0.13847001714669194*filtered_df2.iloc[i,9] + 0.10788809117422904*filtered_df2.iloc[i,10])
    
plt.plot(filtered_df2['TIME'], filtered_df2['bikes'])
plt.plot(filtered_df2['TIME'], reg_list, alpha = 0.7)
plt.title("include 1 week, month and year ago")
plt.show()




















