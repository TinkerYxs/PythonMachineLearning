# This Python file uses the following encoding: utf-8

# 岭回归

import mglearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
X,y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
ridge = Ridge().fit(X_train,y_train)
print("Traing set score:{:.2f}".format(ridge.score(X_train,y_train)))#0.95
print("Test set score:{:.2f}".format(ridge.score(X_test,y_test)))