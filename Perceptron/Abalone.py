# -*- coding = utf-8 -*-
# @Time :2022/3/20 16:31
# @Author :彭程
# @File :Abalone.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
import numpy as np

data_file = 'A30.csv'
df = pd.read_csv(data_file)
df.columns = [
"gender","length","diameter","height","fullweight","fishweight","visceralweight","shellweight","rings"
]
a = df.gender.value_counts()
print(a)
data = np.array(df.iloc[:60, [0, 1, 3]])     #在这里我选择的是csv表格中的0，1，3列，对应鲍鱼的种类和其他两种描述数据
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

plt.scatter(df[0:31]['length'], df[0:31]['height'], label='M')
plt.scatter(df[32:61]['length'], df[32:61]['height'],label='I')
plt.xlabel('length')
plt.ylabel('height')
plt.legend()
plt.show()
