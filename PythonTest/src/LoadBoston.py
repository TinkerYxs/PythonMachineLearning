# This Python file uses the following encoding: utf-8
# 波士顿房价数据集

from sklearn.datasets import load_boston
import mglearn
boston = load_boston()
print("Data shape:{}".format(boston.data.shape))

X,y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))