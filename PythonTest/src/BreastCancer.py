# This Python file uses the following encoding: utf-8
# 威斯康辛州乳腺癌数据集


import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier

cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))

print("Shape of cancer data: {}".format(cancer.data.shape))

print("Sample counts per class:\n{}".format(
        {n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))


#对比不能k值的k近邻情况下的训练精度和测试精度
X_train,X_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,stratify=cancer.target,random_state=66)
training_accuracy = []
test_accuracy =[]
#n_neighbors取值从1到10
neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
    #构建模型
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    #记录训练集精度
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
plt.plot(neighbors_settings,test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend();
plt.show();
    