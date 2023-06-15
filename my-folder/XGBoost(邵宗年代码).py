# _*_coding : UTF-8 _*_
# @Time : 2023/6/8 20:05
# @Author : NEFU_GaoMY
# @File : XGBoost
# @Project : pythonProject
# import os
import pandas as pd
# import numpy as np
from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import sklearn.metrics
import xgboost as xgb
data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\train.csv')  # 读取csv文件
x = data.iloc[:,0:97]# 逗号前表示行，逗号后表示列
y = data.iloc[:,-1]
y_pre_list=[]
index_l=[]
print(x)
print(y)
###########
k = 10
cv = KFold(n_splits=k, shuffle=True)
for train_index, test_index in cv.split(data):
    x_train = pd.DataFrame(data.iloc[train_index, 0:97])
    x_test = pd.DataFrame(data.iloc[test_index, 0:97])
    y_train = pd.DataFrame(data.iloc[train_index, -1])
    y_test = pd.DataFrame(data.iloc[test_index, -1])
    xg_classifier = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.4, learning_rate=0.5,
                                      max_depth=5, alpha=10, n_estimators=1000)
#####################################################################################
xgb.XGBClassifier(learning_rate=0.1,
                  n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                  max_depth=6,               # 树的深度
                  min_child_weight = 1,      # 叶子节点最小权重
                  gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                  subsample=0.8,             # 随机选择80%样本建立决策树
                  colsample_btree=0.8,       # 随机选择80%特征建立决策树
                  objective='multi:softmax', # 指定损失函数
                  scale_pos_weight=1,        # 解决样本个数不平衡的问题
                  random_state=20,            # 随机数
                  num_class=8
                  )
xg_classifier.fit(x_train, y_train)
print(xg_classifier.score(x_test, y_test))
y_pre=xg_classifier.predict(x_test)
#精度评价
kappa=sklearn.metrics.cohen_kappa_score(y_test, y_pre,  labels=None, weights=None, sample_weight=None)
OA=accuracy_score(y_test, y_pre)
print("kappa:",kappa,"OA:",OA)
from sklearn.metrics import classification_report#导入混淆矩阵对应的库
print(classification_report(y_test,xg_classifier.predict(x_test)))


