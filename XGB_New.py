# _*_coding : UTF-8 _*_
# @Time : 2023/6/8 23:19 
# @Author : NEFU_GaoMY
# @File : XGB_New 
# @Project : pythonProject
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import sklearn
from sklearn.metrics import confusion_matrix
data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\train.csv')  # 读取csv文件
x = data.iloc[:, 0:97]  # 逗号前表示行，逗号后表示列
y = data.iloc[:, -1]

x_train = pd.DataFrame(x)  # 使用全部数据作为训练集
y_train = pd.DataFrame(y)  # 使用全部数据作为训练集

xg_classifier = xgb.XGBClassifier(learning_rate=0.1,
                                 n_estimators=1000,
                                 max_depth=6,
                                 min_child_weight=1,
                                 gamma=0.,
                                 subsample=1,
                                 colsample_btree=1,
                                 objective='multi:softmax',
                                 scale_pos_weight=1,
                                 random_state=20,
                                 num_class=8)
xg_classifier.fit(x_train, y_train)

# 验证集数据
test_data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\test.csv')  # 读取验证集数据
x_test = test_data.iloc[:, 0:97]
y_test = test_data.iloc[:, -1]

y_pred = xg_classifier.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
kappa = sklearn.metrics.cohen_kappa_score(y_test, y_pred, labels=None, weights=None, sample_weight=None)
print("Confusion Matrix:")
print(conf_matrix)
OA = accuracy_score(y_test, y_pred)
print("kappa:", kappa, "OA:", OA)
print(classification_report(y_test, y_pred))

