#生成数据集
import mglearn

import matplotlib.pyplot as plt


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


