#knn分类
from sklearn.neighbors import  KNeighborsClassifier
#KNN回归
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import os
import pandas as pd
import csv
from  sklearn.datasets import  load_digits
import  matplotlib.pyplot as plt
from  sklearn.model_selection import  train_test_split
from  sklearn.preprocessing import StandardScaler
import  sklearn.preprocessing
from  sklearn.metrics import accuracy_score
from  sklearn.svm import SVC
from  sklearn.naive_bayes import  MultinomialNB
from  sklearn.tree import DecisionTreeClassifier
#加载数据
digits=load_digits()
data=digits.data
#数据探索
print(data.shape)
print(digits.images[0])
#查看第一幅图代表的数字含义
print(digits.target[0])
#将第一幅图显示出来

plt.gray()
plt.imshow(digits.images[0])
plt.show()
#将数据集分隔
train_x,test_x,train_y,test_y=train_test_split(data,digits.target,test_size=0.25)
#采用z-score规范化
ss=sklearn.preprocessing.StandardScaler()
train_ss_x=ss.fit_transform(train_x)
test_ss_x=ss.transform(test_x)
#创建knn分类器
knn=KNeighborsClassifier()
knn.fit(train_ss_x,train_y)
predict_y=knn.predict(test_ss_x)
print('KNN准确率：%.4lf'% accuracy_score(test_y,predict_y))
#创建svm分类器
svm=SVC(gamma='auto')
svm.fit(train_ss_x,train_y)
predict_y=svm.predict(test_ss_x)
print('SVM准确率：%0.4lf'% accuracy_score(test_y,predict_y))
#采用min-max规范化
mm=sklearn.preprocessing.MinMaxScaler()
train_mm_x=mm.fit_transform(train_x)
test__mm_x=mm.transform(test_x)
#创建naive bayes分类器
mnb=MultinomialNB()
mnb.fit(train_mm_x,train_y)
predict_y=mnb.predict(test__mm_x)
print('多项式朴素贝叶斯准确率：%4lf'% accuracy_score(test_y,predict_y))
#创建CART决策树分类器
dct=DecisionTreeClassifier()
dct.fit(train_mm_x,train_y)
predict_y=dct.predict(test__mm_x)
print('CART决策树准确率: %4.lf'% accuracy_score(test_y,predict_y))


