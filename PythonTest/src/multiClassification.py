# This Python file uses the following encoding: utf-8
## 用于多分类的线性模型


## 一对其余 one-vs.-rest
###应用在一个简单的三分类数据集上，用到了一个二维数据集

from sklearn.datasets import make_blobs
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm.classes import LinearSVC
from Lib.idlelib.colorizer import color_config

X,y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0","Class 1","Class 2"])
plt.show()

##在这个数据集上训练一个LinearSVC分类器
linear_svm = LinearSVC().fit(X,y)
print("Coefficient shape: ",linear_svm.coef_.shape)
print("Intercept shape: ",linear_svm.intercept_.shape)
#Coefficient shape:  (3, 2)
#Intercept shape:  (3,)
#coef_每行包含三个类别之一的系数向量，每列包含某个特征对应的系数值，包含两个特征
#intercept_是一维数组，保存每个类别的截距

#将这3个分类器给出的直线可视化

mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim((-10,8))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0','Class 1','Class 2','Line class 0','Line class 1','Line class 2'],loc=(1.01,0.3))
plt.tight_layout(pad=0.4)
plt.show()


##下面例子给出了二维空间中所有区域的预测结果
mglearn.plots.plot_2d_classification(linear_svm,X,fill=True,alpha=.7)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0','Class 1','Class 2','Line class 0','Line class 1','Line class 2'],loc=(1.01,0.3))
plt.tight_layout(pad=0.4)
plt.show()
