# -*- coding: cp936 -*-
# 绪论案例:Boston房价

# %matplotlib inline

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

boston_dataset = datasets.load_boston()
X_full = boston_dataset.data
Y = boston_dataset.target
print(X_full.shape)    # (506, 13)
print(Y.shape)         # (506,)

print(boston_dataset.DESCR)

# 特征选择
selector = SelectKBest(f_regression,k=1)
selector.fit(X_full,Y)
X = X_full[:,selector.get_support()]
print(X.shape)         # (506, 1)

plt.scatter(X,Y,color='black')
plt.show()

# 线性回归
regressor = LinearRegression(normalize=True)
regressor.fit(X,Y)

plt.scatter(X,Y,color='black')
plt.plot(X,regressor.predict(X),color='red',linewidth=3)
plt.show()

# SVM
regressor = SVR()
regressor.fit(X,Y)

plt.scatter(X,Y,color='black')
plt.scatter(X,regressor.predict(X),color='red',linewidth=3)
plt.show()

# Random Forest回归
regressor = RandomForestRegressor()
regressor.fit(X,Y)

plt.scatter(X,Y,color='black')
plt.scatter(X,regressor.predict(X),color='red',linewidth=3)
plt.show()