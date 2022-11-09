# -*- coding = utf-8 -*-
# @Time :2022/5/12 16:11
# @Author :彭程
# @File :Svm.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score  # 返回正确的比例
from sklearn.preprocessing import LabelEncoder

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.close('all')


def main():
    # 1.获取x,y
    data = pd.read_table('watermelon30a.txt', delimiter=',')
    x = pd.DataFrame({'密度': data['密度'], '含糖率': data['含糖率']})
    x = x.values.tolist()
    encoder = LabelEncoder()  # 将好瓜坏瓜映射为1/0
    y = encoder.fit_transform(data['好瓜']).tolist()
    x, y = np.array(x), np.array(y)
    # 2.1.线性核
    linear_svm = svm.SVC(C=0.5,  # 惩罚参数
                         kernel='linear')
    linear_svm.fit(x, y)
    y_pred = linear_svm.predict(x)
    print('**linear_svm的准确率**: %s' % (accuracy_score(y_pred=y_pred, y_true=y)))
    ##2.2.高斯核
    gauss_svm = svm.SVC(C=0.5, kernel='rbf')
    gauss_svm.fit(x, y)
    y_pred2 = gauss_svm.predict(x)
    print('**gauss_svm的准确率**: %s' % (accuracy_score(y_pred=y_pred2, y_true=y)))
    class_method = {'线性核': linear_svm, '高斯核': gauss_svm}
    visual(data, class_method)


##数据特征可视化
def visual(data, class_method):
    colormap = dict(zip(data['好瓜'].value_counts().index.tolist(), ['blue', 'green']))  # 坏瓜好瓜颜色
    die = data.groupby('好瓜')
    plt.figure()
    for species, klass in die:
        plt.scatter(klass['密度'], klass['含糖率'],
                    color=colormap[species],
                    label=species
                    )
    for name, model in class_method.items():
        sv = model.support_vectors_
        plt.plot(sv[0], sv[1], label=str(name) + '_supported_vector')
    plt.legend(frameon=True, title='好瓜', loc="upper left")
    plt.title('SVC')
    plt.show()


if __name__ == "__main__":
    main()