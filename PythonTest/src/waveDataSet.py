# This Python file uses the following encoding: utf-8
# wave数据集K近邻回归

import mglearn

import numpy as np

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from Lib.random import random
from networkx.utils.decorators import random_state
from IPython.core.pylabtools import figsize

#3个测试点，一个近邻
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

#3个近邻  预测结果为3个近邻的平均值
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

#用于回归的K近邻算法
from sklearn.neighbors import KNeighborsRegressor
X,y =mglearn.datasets.make_wave(n_samples=40)
#将Wave数据集分为训练集和测试集
X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=0)
#模型实例化，并将邻居个数设为3
reg=KNeighborsRegressor(n_neighbors=3)
#利用训练数据和训练目标值来拟合模型
reg.fit(X_train,y_train)

#对测试集进行预测
print("Test set predictions:\n{}".format(reg.predict(X_test)))
#用score方法评估模型
print("Test set R^2:{:.2f}".format(reg.score(X_test,y_test)))



#分析KNeighborsRegressor
#创建一个由许多点组成的测试数据集
fig,axes = plt.subplots(1,3,figsize=(15,4))
#创建1000个数据点，在-3和3之间均匀分布
line = np.linspace(-3,3,1000).reshape(-1,1)

for n_neighbors,ax in zip([1,3,9],axes):
    reg =KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'^',c=mglearn.cm2(1),markersize=8)
    ax.set_title(
        "{} neighbors(s)\n train score:{:.2f} test score:{:.2f}".format(
            n_neighbors,reg.score(X_train,y_train),reg.score(X_test,y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model Predictions","Training data/target","Test data/target"],loc="best")

plt.show()