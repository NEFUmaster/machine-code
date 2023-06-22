# _*_coding : UTF-8 _*_
# @Time : 2023/6/8 20:05
# @Author : NEFU_GaoMY
# @File : RFECV递归
# @Project : pythonProject

import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

# 1.读取训练数据集
file_path=r"E:\jupyter\大兴安岭数据处理\train.csv"
data=pd.read_csv('train.csv')
x= data.iloc[:,0:97]
y= data.iloc[:,-1]
print(x)
print(y)

# 1.标准化处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x)

# 2.构建RF模型
RFC_ = RFC()                               # 随机森林
c= RFC_.fit(x, y).feature_importances_    # 特征重要性
print("重要性：")
print(c)

# 3. 交叉验证递归特征消除法
selector = RFECV(RFC_, step=1, cv=10)       # 采用交叉验证，每次排除一个特征，筛选出最优特征
selector = selector.fit(x, y.astype('int'))
x_wrapper = selector.transform(x)          # 最优特征
score =cross_val_score(RFC_ , x_wrapper, y, cv=10).mean()   # 最优特征分类结果
print(score)
print("最佳数量和排序")
print(selector.support_)                                    # 选取结果
print(selector.n_features_)                                 # 选取特征数量
print(selector.ranking_)# 依次排数特征排序

# 4.递归特征消除法
selector1 = RFE(RFC_, n_features_to_select=31, step=1).fit(x, y.astype('int'))      # n_features_to_select表示筛选最终特征数量，step表示每次排除一个特征
selector1.support_.sum()
print(selector1.ranking_)                                             # 特征排除排序
print(selector1.n_features_)                                          # 选择特征数量
x_wrapper1 = selector1.transform(x)                                   # 最优特征
score =cross_val_score(RFC_, x_wrapper, y, cv=9).mean()
print(score)

# 5.递归特征消除法和曲线图选取最优特征数量
score = []                                                            # 建立列表
for i in range(1, 97, 1):
    x_wrapper = RFE(RFC_, n_features_to_select=i, step=1).fit_transform(x, y)    # 最优特征
    once = cross_val_score(RFC_, x_wrapper, y, cv=9).mean()                      # 交叉验证
    score.append(once)                                                           # 交叉验证结果保存到列表
print(max(score), (score.index(max(score))*1)+1)                                 # 输出最优分类结果和对应的特征数量
print(score)
plt.figure(figsize=[97, 32])
plt.plot(range(1, 97, 1), score)
plt.xticks(range(1, 97, 1))
plt.show()