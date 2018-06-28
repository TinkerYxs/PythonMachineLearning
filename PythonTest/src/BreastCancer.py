# This Python file uses the following encoding: utf-8
# 威斯康辛州乳腺癌数据集

import numpy as np

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))

print("Shape of cancer data: {}".format(cancer.data.shape))

print("Sample counts per class:\n{}".format(
        {n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))