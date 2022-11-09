# -*- coding = utf-8 -*-
# @Time :2022/4/14 16:59
# @Author :彭程
# @File :Fruit.py
# @Software: PyCharm


import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics, neighbors
from sklearn.neighbors import KNeighborsClassifier

print("----------------------------原始语料处理--------------------------------")
#txt转csv
with open('fruit.csv', 'w+', newline='') as csvfile:
    spamriter = csv.writer(csvfile, dialect='excel')
    with open('fruit_data.txt', 'r',encoding='utf-8') as filein:
        for line in filein:
            line_list = line.strip('\n').split(',')
            spamriter.writerow(line_list)

data_file = 'fruit.csv'
Fruit = pd.read_csv(data_file)
print(Fruit)

print("----------------------------数据类别计数--------------------------------")
# 数据类别计数
Fruit.columns = ["types", "mass", "width", "height", "color_score"]
counts = Fruit.types.value_counts()

def create_data():
    # #数据描述
    # sns.pairplot(Fruit, hue='types')
    # plt.show()
    #数据类别计数
    Fruit.columns = ["types", "mass", "width", "height", "color_score"]
    counts = Fruit.types.value_counts()
    print(counts)
create_data()

print("----------------------------划分训练集--------------------------------")
X = Fruit[["mass", "width", "height", "color_score"]]
y = Fruit["types"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("————————训练集详情———————")
print(y_train.value_counts())
print("————————测试集详情———————")
print(y_test.value_counts())

print("----------------------------KNN算法训练模型--------------------------------")
clf_sk = KNeighborsClassifier(n_neighbors=3) #1
clf_sk.fit(X_train, y_train)
accuary = clf_sk.score(X_test, y_test)
print("测试集评估得分",accuary)

print("----------------------------预测数据集--------------------------------")
target = [[192,8.4,7.3,0.55],[200,7.3,10.5,0.72]]
predict_target = clf_sk.predict(target)
print("测试结果为",predict_target)