# _*_coding : UTF-8 _*_
# @Time : 2023/6/8 20:05
# @Author : NEFU_GaoMY
# @File : 递归消除
# @Project : pythonProject

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 导入数据
dataset = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\train.csv')

# 准备训练数据
X = dataset.iloc[:, :-5].values  # 自变量
y = dataset.iloc[:, -1].values  # 因变量

# 创建分类模型
estimator = RandomForestClassifier(n_estimators=150, random_state=1)

# 特征选择
selector = RFE(estimator, n_features_to_select=30)
selector = selector.fit(X, y)

selected_features = dataset.columns[:-5][selector.support_]
# 输出特征选择结果
print(selector.support_)      # 哪些特征入选最后特征，True 表示入选
print(selector.ranking_)      # 每个特征的得分排名，特征得分越低（1最好），表示特征越好
print(selector.n_features_)   # 挑选了几个特征
print(selected_features)