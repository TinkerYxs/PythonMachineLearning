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


#Ridge 有个alpha 参数 以上默认是1.0，alpha 越大限制性越大，alpha越小 越接近线性回归

ridge10 = Ridge(alpha=10).fit(X_train,y_train)
print("Traing set score:{:.2f}".format(ridge10.score(X_train,y_train)))
print("Test set score:{:.2f}".format(ridge10.score(X_test,y_test)))


ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("Traing set score:{:.2f}".format(ridge01.score(X_train,y_train)))
print("Test set score:{:.2f}".format(ridge01.score(X_test,y_test)))


###画图看L2 正则化的影响 
import matplotlib.pyplot as plt
plt.plot(ridge.coef_,'s',label="Ridge alpha=1")
plt.plot(ridge10.coef_,'^',label="Ridge alpha=10")
plt.plot(ridge01.coef_,'v',label="Ridge alpha=0.1")


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train,y_train)
plt.plot(lr.coef_,'o',label="LinearRegression")

plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()

plt.show()

##学习曲线
mglearn.plots.plot_ridge_n_samples()
plt.show()