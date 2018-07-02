# This Python file uses the following encoding: utf-8


import mglearn


import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# wave 数据集线性模型
mglearn.plots.plot_linear_regression_wave()#线性模型对数据集的预测结果

plt.show()


#线性回归，最小二乘法
from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

lr = LinearRegression().fit(X_train,y_train)

print("lr.coef_:{}".format(lr.coef_))#斜率参数
print("lr.intercept_:{}".format(lr.intercept_))#截距参数

print("Traing set score:{:.2f}".format(lr.score(X_train,y_train)))#0.67
print("Test set score:{:.2f}".format(lr.score(X_test,y_test)))#0.66 相差不大 欠拟合，没有过拟合

#波士顿房价数据集上的线性回归

X,y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
lr = LinearRegression().fit(X_train,y_train)
print("Traing set score:{:.2f}".format(lr.score(X_train,y_train)))#0.95
print("Test set score:{:.2f}".format(lr.score(X_test,y_test)))#0.61  过拟合



































