# -*- coding = utf-8 -*-
# @Time :2022/4/1 17:43
# @Author :彭程
# @File :sample.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np

data_file = '企鹅.xlsx'
df = pd.read_excel(data_file)
df.columns = [
    'species', 'island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','body_mass_g','sex'
]
a = df.species.value_counts()
print(a)
data = np.array(df.iloc[:200, [0, 2, 3]])
X, y = data[:, [1,2]], data[:,0]




def per(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = Perceptron(fit_intercept=True,
                     max_iter=1000,
                     shuffle=True)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    print(clf.coef_)   #w
    print(clf.intercept_)   #截距

    predict_y = clf.predict(X_test)
    print(metrics.classification_report(y_test, predict_y))
    print("感知机分类正确率: ", metrics.accuracy_score(y_test, predict_y))

    # x_ponits = np.arange(30, 55)
    # y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
    # plt.plot(x_ponits, y_)
    #
    # plt.scatter(df[:50]['culmen_length_mm'], df[:50]['culmen_depth_mm'], label='Adelie')
    # plt.scatter(df[160:200]['culmen_length_mm'], df[160:200]['culmen_depth_mm'],label='Chinstrap')
    # plt.xlabel('culmen_length_mm')
    # plt.ylabel('culmen_depth_mm')
    # plt.legend()
    # plt.show()


def logistic(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    print(clf.coef_, clf.intercept_)


    predict_y = clf.predict(X_test)
    print(metrics.classification_report(y_test, predict_y))
    print("逻辑回归分类正确率: ", metrics.accuracy_score(y_test, predict_y))


    x_ponits = np.arange(30, 55)
    y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_ponits, y_)

    plt.scatter(df[:50]['culmen_length_mm'], df[:50]['culmen_depth_mm'], label='Adelie')
    plt.scatter(df[160:200]['culmen_length_mm'], df[160:200]['culmen_depth_mm'], label='Chinstrap')
    plt.xlabel('culmen_length_mm')
    plt.ylabel('culmen_depth_mm')
    plt.legend()
    plt.show()

per(X,y)
logistic(X,y)