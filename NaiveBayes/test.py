# -*- coding = utf-8 -*-
# @Time :2022/5/4 14:42
# @Author :彭程
# @File :test.py
# @Software: PyCharm
import csv
import math
from itertools import chain

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split

print("----------------------------原始素材预处理--------------------------------")
# data_file = '2.0.csv'
# watermelon = pd.read_csv(data_file,encoding='GBK')
# print(type(watermelon))
# print(watermelon)
# # 首先将pandas读取的数据转化为array
# data_array = np.array(watermelon)
# # 然后转化为list形式
# Watermelon =data_array.tolist()
# print(Watermelon)
data = open(r'2.0.csv')
reader = csv.reader(data)#采用csv.reader读取文件
for row in reader:#reader不能直接使用，需要通过循环提取每一行的数据
    headers=row
    break#只需要将属性提取出来
Watermelon = []
labelList = []
for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)):
        rowDict[headers[i]] = row[i]
    Watermelon.append(rowDict)
print(Watermelon)
vec = DictVectorizer()  #实例化
dummyX = vec.fit_transform(Watermelon).toarray()#输出转化后的特征矩阵
print("-----------------------------dummyX-------------------------------")
print(dummyX)
lb = preprocessing.LabelBinarizer()
print("-----------------------------dummyY-------------------------------")
dummyY1 = lb.fit_transform(labelList)## 将标签矩阵二值化，‘是’-1，‘否’-0
dummyY = list(chain.from_iterable(dummyY1)) #将标签矩阵变为一维矩阵
print(dummyY)
print("------------------------------------------------------------")
print(vec.get_feature_names_out())


print("----------------------------朴素贝叶斯算法训练模型--------------------------------")
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(dummyX,dummyY,test_size=5,random_state=0)
clf = GaussianNB()
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
print("算法得分为:",score) 