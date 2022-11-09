# -*- coding = utf-8 -*-
# @Time :2022/5/25 14:45
# @Author :彭程
# @File :test2.py
# @Software: PyCharm

import classifier as classifier
import pandas as pd
from pandas import read_csv
from sklearn import datasets
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


#中文不乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("***********************原始数据集的导入***********************")
Abalone_data = pd.read_csv("abalone.csv")
print("-------------特征定义-------------")
Abalone_data.info()
print(Abalone_data)
print("-------------样本规模-------------")
print(Abalone_data.shape)
print("-------------检查样本是否有缺失值-------------")
print("缺失值数量：\n",Abalone_data.isnull().sum())
print("缺失值占比：\n",Abalone_data.isna().mean())


print("***********************数据预处理***********************")
print("-------------将性别这一属性进行重命名-------------")
Abalone_data = Abalone_data.rename(columns={'gender':'Class'})
print(Abalone_data.head(5))
print("-------------样本数据的基本统计信息-------------")
print(Abalone_data.describe())
print("-------------重复值查看与处理-------------")
flag = Abalone_data.duplicated(keep=False)
print(flag)
repeat = flag[flag.values==True].index
print("重复行的索引:\n",repeat)
print("-------------特征值数值化处理-------------")
Abalone_data_Class = {'M':1,'F':0,'I':2}
Abalone_data['Class'] =Abalone_data['Class'].map(Abalone_data_Class)

print("***********************特征分析***********************")
print("-------------各特征分布分析-------------")
print(Abalone_data.groupby('length').size())
print(Abalone_data.groupby('diameter').size())
print(Abalone_data.groupby('height').size())
print(Abalone_data.groupby('fullweight').size())
print(Abalone_data.groupby('fishweight').size())
print(Abalone_data.groupby('visceralweight').size())
print(Abalone_data.groupby('shellweight').size())
print(Abalone_data.groupby('rings').size())
print("-------------目标值分布分析-------------")
print(Abalone_data['Class'].value_counts(1))
print("-------------数值型征分布分析-------------")
Abalone_data.hist()
pyplot.show()

print("***********************模型预测与评估***********************")
print("-------------划分训练集与测试集-------------")
Abalone_data2 = Abalone_data.copy()
Abalone_data2.drop(['Class'],axis=1,inplace=True)
Abalone_data2.head()
x = Abalone_data2
y =Abalone_data['Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.70)
print(x_train.shape)
print(x_test.shape)
print("-------------数据预处理标准化------------")
std = StandardScaler()
x_train=std.fit_transform(x_train)
x_test=std.fit_transform(x_test)
print("-------------保存标准化后数据集------------")
Abalone_data.to_csv("Abalone_stand.csv",header = True,index = False)


print("***********************逻辑回归模型(LogisticRegression)***********************")
#训练逻辑回归模型
lg = LogisticRegression(max_iter=500)
lg.fit(x_train, y_train)
y_score1 = lg.fit(x_train,y_train).predict_proba(x_test)
# print("---------------")
# print(y_score1)
# print(type(y_score1))
print(lg.coef_)#权重参数
y_predict1 = lg.predict(x_test)
#逻辑回归模型测试结果
print("准确率：", lg.score(x_test, y_test))
print("召回率：")
print(classification_report(y_test, y_predict1, labels=[1, 0, 2], target_names=["M", "F","I"]))
print("-------------逻辑回归-混淆矩阵------------")
predict_y = lg.predict(x_test)
confusion_ma1 = confusion_matrix(y_test, predict_y)
print(confusion_ma1)
plt.matshow(confusion_ma1)
plt.title('逻辑回归-混淆矩阵')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("***********************支持向量机模型(SVM)***********************")
classifier2 = svm.SVC(C=1,kernel='linear').fit(x_train,y_train)
data_train_result =classifier2.predict(x_train)
data_test_result = classifier2.predict(x_test)

y_score2 = classifier2.decision_function(x_test)

print('decision_function:\n', classifier2.decision_function(x_train))
print('\npredict:\n', classifier2.predict(x_test))

#支持向量机模型测试结果
print("\nlinear线性核函数-训练集 准确率：",classifier2.score(x_train,y_train))
print("\nlinear线性核函数-测试集 准确率：",classifier2.score(x_test,y_test))

print( '\n召回率：\n',classification_report(y_test, data_test_result, labels=[1, 0, 2], target_names=["M", "F","I"]))
print("-------------SVM 混淆矩阵------------")
conf_matrix2 = confusion_matrix(y_test, data_test_result)
print(conf_matrix2)
plt.matshow(conf_matrix2)
plt.title('SVM-混淆矩阵')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("***********************K近邻（KNN）***********************")
classifier3 = KNeighborsClassifier()
classifier3.fit(x_train, y_train)

#K近邻（KNN）模型测试结果
print("准确率：",classifier3.score(x_test, y_test))

print('标签值预测：')
y_predict3 = classifier3.predict(x_test)
print(y_predict3)

y_score3 = classifier3.fit(x_train, y_train).predict_proba(x_test)
print('\n概率预测：\n',y_score3)
print( '\n召回率:\n',classification_report(y_test, y_predict3, labels=[1, 0, 2], target_names=["M", "F","I"]))
print("-------------KNN 混淆矩阵------------")
conf_matrix3 = confusion_matrix(y_test, y_predict3)
print(conf_matrix3)
plt.matshow(conf_matrix3)
plt.title('KNN-混淆矩阵')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("***********************贝叶斯(Bayes)***********************")
classifier4 = GaussianNB()
classifier4.fit(x_train, y_train)
y_predict4 = classifier4.predict(x_test)

#预测分析
print("分类准确率：",classifier4.score(x_test, y_test))
print("\n分类报告:\n ", classification_report(y_test, y_predict4, labels=[1, 0, 2], target_names=["M", "F","I"]))
y_score4 = classifier4.fit(x_train,y_train).predict_proba(x_test)
print("\n预测概率:\n ",y_score4)

print("-------------贝叶斯-混淆矩阵------------")
conf_matrix4 = confusion_matrix(y_test, y_predict4)
print(conf_matrix4)
plt.matshow(conf_matrix4)
plt.title('贝叶斯-混淆矩阵')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("***********************随机森林(RandomForest)***********************")
#随即森林分类函数，设置决策树数目为100，CART树做划分时对特征的评价标准选择信息增益
rfc = RandomForestClassifier(n_estimators=100,criterion='entropy')
rfc = rfc.fit(x_train,y_train)
y_score5 = rfc.predict_proba(x_test)
y_predict5 = rfc.predict(x_test)

#预测分析
print("分类正确率: ", rfc.score(x_test,y_test))
print("\n分类报告:\n ", classification_report(y_test, y_predict5, labels=[1, 0, 2], target_names=["M", "F","I"]))
print("\n预测概率:\n ",y_score5)

print("-------------随机森林-混淆矩阵------------")
conf_matrix5 = confusion_matrix(y_test, y_predict5)
print(conf_matrix5)
plt.matshow(conf_matrix5)
plt.title('随机森林-混淆矩阵')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("***********************决策树***********************")
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train,y_train)
y_predict6 = clf.predict(x_test)
y_score6 = clf.predict_proba(x_test)

#预测分析
print("分类正确率: ", clf.score(x_test,y_test))
print("\n分类报告:\n ", classification_report(y_test, y_predict6, labels=[1, 0, 2], target_names=["M", "F","I"]))
print("\n预测概率:\n ",y_score6)

print("-------------决策树-混淆矩阵------------")
conf_matrix6 = confusion_matrix(y_test, y_predict6)
print(conf_matrix6)
plt.matshow(conf_matrix6)
plt.title('决策树-混淆矩阵')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


