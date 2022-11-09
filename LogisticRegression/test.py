# -*- coding = utf-8 -*-
# @Time :2022/4/1 16:14
# @Author :彭程
# @File :test.py
# @Software: PyCharm

import inline as inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as inline
from sklearn.linear_model import Perceptron
import os
from math import exp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#导入数据
data_file = 'A30.csv'
Abalone = pd.read_csv(data_file)
# print(Abalone)
# 数据类别计数
Abalone.columns = ["gender", "length", "diameter", "height", "fullweight", "fishweight", "visceralweight","shellweight", "rings"]
counts = Abalone.gender.value_counts()

def create_data():
    #数据描述
    sns.pairplot(Abalone, hue='gender')
    plt.show()
    #数据类别计数
    Abalone.columns = [ "gender","length","diameter","height","fullweight","fishweight","visceralweight","shellweight","rings"]
    counts = Abalone.gender.value_counts()
    print(counts)
    # class_dict = {"M": 0, "I": 1}
    # Abalone["Class"] = Abalone["Class"].map(class_dict)
    # Abalone["Class"].value_counts()
create_data()

def logistic(Abalone):
    #划分训练集和测试集
    X = Abalone[["length", "height"]]
    y = Abalone["gender"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(y_train.value_counts())
    print(y_test.value_counts())

    #构建分类模型
    classifier = LogisticRegression(C=1e3, solver='lbfgs')
    classifier.fit(X_train, y_train)

    #性能评估
    predict_y = classifier.predict(X_test)
    print(metrics.classification_report(y_test, predict_y))
    print("分类正确率: ", metrics.accuracy_score(y_test, predict_y))

    混淆矩阵热点图
    colorMetrics = metrics.confusion_matrix(y_test, predict_y)
    sns.heatmap(colorMetrics, annot=True, fmt='d')
    plt.show()

    #模型系数
    # coef_df = pd.DataFrame(classifier.coef_, columns=Abalone.columns[0:2])
    # coef_df.round(1)
    # coef_df["intercept"] = classifier.intercept_
    # coef_df.round(1)
logistic(Abalone)


def Perceptron1(Abalone):
    data = np.array(Abalone.iloc[:60, [0, 1, 3]])     #在这里我选择的是csv表格中的0，1，3列，对应鲍鱼的种类和其他两种描述数据
    X, y = data[:, [1,2]], data[:,0]
    clf = Perceptron(fit_intercept=True,
                     max_iter=1000,
                    shuffle=True)
    clf.fit(X, y)
    print(clf.coef_)
    print(clf.intercept_)

    x_ponits = np.arange(0.30, 0.55)
    y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
    plt.plot(x_ponits, y_)

    plt.scatter(Abalone[0:31]['length'], Abalone[0:31]['height'], label='M')
    plt.scatter(Abalone[32:61]['length'], Abalone[32:61]['height'],label='I')
    plt.xlabel('length')
    plt.ylabel('height')
    plt.legend()
    plt.show()
Perceptron1(Abalone)