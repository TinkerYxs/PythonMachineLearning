# This Python file uses the following encoding: utf-8
##将Logistic回归和线性SVM linearSVC 支持向量分类器 模型 应用到forge数据集上
##并将线性模型找到的决策边界可视化
##两个模型默认使用L2正则化
##这两个模型决定正则化强度的是权衡参数C，C越大，正则化越弱

from sklearn.linear_model import LogisticRegression##并不是回归
from sklearn.svm import LinearSVC
import mglearn
import matplotlib.pyplot as plt


X,y = mglearn.datasets.make_forge()
fig,axes = plt.subplots(1, 2, figsize=(10,3))

for model, ax in zip([LinearSVC(),LogisticRegression()],axes):
    clf = model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=False,eps=0.5,ax=ax,alpha=.7)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    
    ax.set_title("{}".format(clf.__class__.__name__))##这里有错 找不出来，clf._class_._name_###注意是双2下划线获取函数名
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()


##不同C值的linearSVC图示
mglearn.plots.plot_linear_svc_regularization()
plt.show()



#在乳腺癌数据集上详细分析LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,stratify=cancer.target,random_state=42)
##C默认是1
logreg = LogisticRegression().fit(X_train,y_train)
print("Traing set score:{:.3f}".format(logreg.score(X_train,y_train)))
print("Test set score:{:.3f}".format(logreg.score(X_test,y_test)))

##C=100
logreg100 = LogisticRegression(C=100).fit(X_train,y_train)
print("Traing set score:{:.3f}".format(logreg100.score(X_train,y_train)))
print("Test set score:{:.3f}".format(logreg100.score(X_test,y_test)))

#C=0.01
logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)
print("Traing set score:{:.3f}".format(logreg001.score(X_train,y_train)))
print("Test set score:{:.3f}".format(logreg001.score(X_test,y_test)))



#作图比较正则化参数c取三个不同的值时模型学到的系数
plt.plot(logreg.coef_.T,'o',label="C=1")
plt.plot(logreg100.coef_.T,'^',label="C=100")
plt.plot(logreg001.coef_.T,'v',label="C=0.001")
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
plt.hlines(0,0,cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-7,7)
plt.tight_layout(pad=0.4)
#tight_layout命令还有三个关键字参数：pad、w_pad、h_pad。
#pad用于设置绘图区边缘与画布边缘的距离大小
#w_pad用于设置绘图区间水平距离的大小
#h_pad用于设置绘图区间垂直距离的大小
plt.legend()
plt.show()


##不同C值，L1惩罚的logistic回归

for C,maker in zip([0.001,1,100],['o','^','v']):
    lr_l1 = LogisticRegression(C=C,penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}:{:.2f}".format(C,lr_l1.score(X_train,y_train))) 
    print("Test accuracy of l1 logreg with C={:.3f}:{:.2f}".format(C,lr_l1.score(X_test,y_test)))

    plt.plot(lr_l1.coef_.T,maker,label="C={:.3f}".format(C))
    plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
    plt.hlines(0,0,cancer.data.shape[1])
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-7,7)
    plt.tight_layout(pad=0.4)
    plt.legend(loc=3)
plt.show()





