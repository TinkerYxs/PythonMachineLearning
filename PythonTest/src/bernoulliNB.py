# This Python file uses the following encoding: utf-8
# 了解朴素贝叶斯分类器

import numpy as np
from sqlalchemy.sql.expression import false
X = np.array([[0,1,0,1],
             [1,0,1,1],
             [0,0,0,1],
             [1,0,1,0]])
y = np.array([0,1,0,1])


counts={}
for label in np.unique(y):
    print(label)
    #对每个类别进行遍历
    #计算(求和)每个特征中1的个数
    print(X[False])
    print(X[True])
    print(y==label)
    print(X[y==label])
    #计算没列中非零元素的个数
    counts[label] = X[y==label].sum(axis=0)#axis=1表示按行相加 , axis=0表示按列相加
print("Feature counts:\n{}".format(counts))
