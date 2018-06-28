#生成数据集
import mglearn

import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize


X,y = mglearn.datasets.make_forge()#forge 是什么意思？

#数据集绘图
mglearn.discrete_scatter(X[:,0],X[:,1], y)
plt.legend(["Class 0","Class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:{}".format(X.shape))
plt.show()

X,y = mglearn.datasets.make_wave(n_samples=40)#wave 是什么意思？
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

#k邻近
mglearn.plots.plot_knn_classification(n_neighbors=1)

plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

#分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
#导入并实例化，设置参数
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
#拟合
clf.fit(X_train,y_train)
#预测
print("Test set predictions: {}".format(clf.predict(X_test)))
#评估模型的泛化能力好坏
print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))
#决策边界可视化,将1个，3个，9个邻居三种情况的决策边界可视化
fig,axes = plt.subplots(1,3,figsize=(10,3))

for n_neighbors,ax in zip([1,3,9],axes):
    #fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()


