from sklearn.datasets import  load_iris




import sys


print("Python version:{}".format(sys.version))

import pandas as pd
print("pandas version:{}".format(pd.__version__))

import matplotlib
print("matplotlib version:{}".format(matplotlib.__version__))

import numpy as np
print ("Numpy version:{}".format(np.__version__))

import scipy as sp
print("Scipy version:{}".format(sp.__version__))

import IPython
print("IPython version:{}".format(IPython.__version__))

import sklearn
print("scikit-learn version:{}".format(sklearn.__version__))
iris_dataset = load_iris()

print(("keys of iris_dataset:\n{}".format(iris_dataset.keys())))

print(iris_dataset['DESCR'][:193]+"\n...")

print("Target names:{}".format(iris_dataset['target_names']))

print("Feature names:\n{}".format(iris_dataset['feature_names']))

print ("Type of data:{}".format(type(iris_dataset['data'])))

print("shape of data:{}".format(iris_dataset['data'].shape))

print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))

print("Type of target:{}".format(type(iris_dataset['target'])))

print("Shape of target:{}".format(iris_dataset['target'].shape))

print("Target:\n{}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split  
X_train,X_test,y_train,y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0
    )
print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))

print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

#利用X_train中的数据创建DataFrame
# 利用iris_dataset.fearure_names中的字符串对数据列进行标记

import mglearn
import matplotlib.pyplot as plt
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按y_train着色
#alpha 透明度
#marker 画图点的形状标记
#figsize画图窗口的大小 宽，长
#c 颜色
#第一个参数是DataFrame
#hist_kwds：(other plotting keyword arguments，可选)，与hist相关的字典参数
#hist 对角线上是直方图
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='+',
                       hist_kwds={'bins':40}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()

from sklearn.neighbors import  KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#KNeighborsClassifier(algorithm='auto',
#                    leaf_size=30,
#                    metric='minkowski',
#                    metric_params=None,
#                    n_neighbors=1,
#                    p=2,
#                    weights='uniform')
 
X_new=np.array([[5,2.9,1,0.2]])

print("X_new.shape:{}".format(X_new.shape))

prediction =knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))


y_pred=knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))
print("Test set score:{:.2f}".format(np.mean(y_pred==y_test)))

print("Test set score:{:.2f}".format(knn.score(X_test,y_test))) 





















