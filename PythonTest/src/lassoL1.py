# This Python file uses the following encoding: utf-8
# L1 正则化 线性回归 Lasso回归
# 扩展在波士顿房价数据集

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import mglearn
import numpy as np

boston = load_boston()

X,y = mglearn.datasets.load_extended_boston()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)#分训练集和测试集

###表现很差 存在欠拟合 只用到了105个特征中的4个
lasso =Lasso().fit(X_train,y_train)##拟合
print("Training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print("Number of features used:{}".format(np.sum(lasso.coef_!=0)))

### 尝试减小alpha 增加max_iter的值
#max_iter 运行迭代的最大次数

#我们增大max_iter的值，否则模型会警告我们，说应该增大max_iter的值
lasso001 = Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
print("Training set score:{:.2f}".format(lasso001.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lasso001.score(X_test,y_test)))
print("Number of features used:{}".format(np.sum(lasso001.coef_!=0)))

lasso00001 = Lasso(alpha=0.0001,max_iter=100000).fit(X_train,y_train)
print("Training set score:{:.2f}".format(lasso00001.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lasso00001.score(X_test,y_test)))
print("Number of features used:{}".format(np.sum(lasso00001.coef_!=0)))



####作图
###画图看L2 正则化的影响 
import matplotlib.pyplot as plt
plt.plot(lasso.coef_,'s',label="Ridge alpha=1")
plt.plot(lasso001.coef_,'^',label="Ridge alpha=0.01")
plt.plot(lasso00001.coef_,'v',label="Ridge alpha=0.0001")


from sklearn.linear_model import Ridge
ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
plt.plot(ridge01.coef_,'o',label="Ridge alpha=0.1")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend(ncol=2,loc=(0,1.05))
plt.ylim(-25,25)


plt.show()














